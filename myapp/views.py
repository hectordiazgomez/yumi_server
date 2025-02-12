import os
import csv
import re
import unicodedata
import time
import threading
import firebase_admin
from firebase_admin import credentials, firestore
import sys
import boto3
import typing as tp
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import pandas as pd
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, Adafactor
from transformers import get_constant_schedule_with_warmup
from django.core.files.base import ContentFile
from sacremoses import MosesPunctNormalizer

def initialize_mpn(lang):
    mpn = MosesPunctNormalizer(lang)
    mpn.substitutions = [
        (re.compile(r), sub) for r, sub in mpn.substitutions
    ]
    return mpn

cred = credentials.Certificate('')
firebase_admin.initialize_app(cred)

db = firestore.client()

session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name=''
)

def save_to_s3(model, tokenizer, path, bucket_name='#Name goes here'): 
    s3 = session.client('s3')  

    model_path = f"models/{path}/pytorch_model.bin"
    tokenizer_path = f"models/{path}/tokenizer_config.json"
    config = f"models/{path}/config.json"
    generation = f"models/{path}/generation_config.json"
    sentence = f"models/{path}/sentencepiece.bpe.model"
    special_tokens = f"models/{path}/special_tokens_map.json"

    model.save_pretrained(f"models/{path}")
    tokenizer.save_pretrained(f"models/{path}")

    s3.upload_file(model_path, bucket_name, model_path)
    s3.upload_file(tokenizer_path, bucket_name, tokenizer_path)
    s3.upload_file(config, bucket_name, config)
    s3.upload_file(generation, bucket_name, generation)
    s3.upload_file(sentence, bucket_name, sentence)
    s3.upload_file(special_tokens, bucket_name, special_tokens)

    os.remove(model_path)
    os.remove(tokenizer_path)
    os.remove(config)
    os.remove(generation)
    os.remove(sentence)
    os.remove(special_tokens)

def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)
    print("Normalization finished")
    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text, mpn):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

def read_csv_file(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    delimiters = [',', '\t']
    
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                with open(file_path, 'r', encoding=encoding) as csvfile:
                    start = csvfile.read(1024)
                    csvfile.seek(0)
                    dialect = csv.Sniffer().sniff(start)
                    dialect.delimiter = delimiter
                    
                    df = pd.read_csv(file_path, encoding=encoding, sep=delimiter, quoting=csv.QUOTE_MINIMAL, dialect=dialect)
                
                if df.shape[1] >= 2:
                    print(f"Successfully read CSV with encoding: {encoding} and delimiter: {repr(delimiter)}")
                    return df
                else:
                    print(f"File read successfully but doesn't have at least two columns. Trying next format...")
            except Exception as e:
                print(f"Failed to read with encoding: {encoding} and delimiter: {repr(delimiter)}. Error: {str(e)}")
                continue
    
    raise ValueError("Failed to read the CSV file with all attempted encodings and delimiters.")

def main_function(request):
    if request.method == 'GET':
        return JsonResponse({"Hola": "API working ok"})

@csrf_exempt
def train_nmt(request):
    if request.method == 'POST':
        source_lang = request.POST.get('sourceLang')
        target_lang = request.POST.get('targetLang')
        similar_lang = request.POST.get('similarLanguage') 
        userUid = request.POST.get('userUid')
        instanceId = request.POST.get('instanceId')
        path = request.POST.get('path')
        mosesPunctNormalizer = request.POST.get('mosesPunctNormalizer')
        batch_size = int(request.POST.get('batchSize', 32))
        max_length = int(request.POST.get('maxLength', 128))
        warmup_steps = int(request.POST.get('warmupSteps', 500))
        training_steps = int(request.POST.get('trainingSteps', 10000))
        learning_rate = float(request.POST.get('learningRate', 1e-4))
        weight_decay = float(request.POST.get('weightDecay', 0.01))

        required_params = ['sourceLang', 'targetLang']
        missing_params = [param for param in required_params if param not in request.POST]
        if missing_params:
            return JsonResponse({'error': f'Missing required parameters: {", ".join(missing_params)}'}, status=400)

        if 'inputFile' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        model_docs = db.collection('models').where('uid', '==', userUid).where('data.path', '==', path).get()
        if not model_docs:
            return JsonResponse({'error': 'Model document not found'}, status=404)
        
        model_doc = model_docs[0]
        doc_ref = model_doc.reference
        instance_query = db.collection('instances').where('instanceId', '==', instanceId).limit(1).get()
        if not instance_query:
            return JsonResponse({'error': 'Instance not found'}, status=404)
        
        instance_doc = instance_query[0]
        instance_ref = instance_doc.reference

        try:
            instance_ref.update({'active': True})
            doc_ref.update({'currentStep': 'Starting preprocessing'})
            uploaded_file = request.FILES['inputFile']
            file_name = default_storage.save('temp.csv', ContentFile(uploaded_file.read()))
            file_path = default_storage.path(file_name)

            try:
                df = read_csv_file(file_path)
                if df.shape[1] < 2:
                    raise ValueError("CSV file must have at least two columns")
                df.columns = ['source', 'target'] + list(df.columns[2:])
            except Exception as e:
                default_storage.delete(file_name)
                return JsonResponse({'error': f'Failed to read CSV file: {str(e)}'}, status=400)

            default_storage.delete(file_name)
            mpn = initialize_mpn(lang=mosesPunctNormalizer)
            df['source'] = df['source'].apply(lambda x: preproc(x, mpn))
            df['target'] = df['target'].apply(lambda x: preproc(x, mpn))
            doc_ref.update({'currentStep': 'Initializing model and tokenizer'})
            tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

            fix_tokenizer(tokenizer, new_lang=source_lang)
            model.resize_token_embeddings(len(tokenizer))
            doc_ref.update({'currentStep': 'Initializing optimizer'})
            optimizer = Adafactor(
                model.parameters(),
                scale_parameter=False,
                relative_step=False,
                lr=learning_rate,
                clip_threshold=1.0,
                weight_decay=weight_decay,
            )
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
            doc_ref.update({'currentStep': 'Starting training'})
            all_losses = []
            model.train()
            for step in range(training_steps):
                source_texts, target_texts = get_batch(df, batch_size)
                tokenizer.src_lang = source_lang
                inputs = tokenizer(
                    source_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                )
                tokenizer.tgt_lang = target_lang
                labels = tokenizer(
                    target_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                )

                outputs = model(**inputs, labels=labels.input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                all_losses.append(loss.item())

                if step % 10 == 0:
                    print(f"Step {step}, Loss: {loss.item()}")
                    doc_ref.update({
                        'currentStep': step,
                        'currentLoss': loss.item(),
                        'allLosses': all_losses 
                    })

            doc_ref.update({'currentStep': 'Saving model'})
            if not os.path.exists(f"models/{path}"):
                os.makedirs(f"models/{path}")
            model.save_pretrained(f"models/{path}")
            tokenizer.save_pretrained(f"models/{path}")

            save_to_s3(model, tokenizer, path, bucket_name='gaiacloud-one')
            doc_ref.update({'currentStep': 'Training completed', 'trainingStatus': 'success', 'allLosses': all_losses})
            response_data = {'message': 'Training completed successfully and model saved to S3'}

            def stop_instance(instance_id):
                try:
                    instance_ref.update({'active': False})
                    
                    time.sleep(5)
                    ec2 = session.client('ec2')
                    ec2.stop_instances(InstanceIds=[instance_id])
                    print(f"Instance {instance_id} has been stopped.")
                except Exception as e:
                    print(f"Error stopping instance {instance_id}: {str(e)}")
                    doc_ref.update({'currentStep': 'Error stopping instance', 'trainingStatus': 'error'})


            threading.Thread(target=stop_instance, args=(instanceId,)).start()
            return JsonResponse(response_data)

        except Exception as e:
            doc_ref.update({'currentStep': 'Error during training', 'trainingStatus': 'error'})
            def stop_instance(instance_id):
                try:
                    instance_ref.update({'active': False})
                    
                    time.sleep(5)
                    ec2 = session.client('ec2')
                    ec2.stop_instances(InstanceIds=[instance_id])
                    print(f"Instance {instance_id} has been stopped.")
                except Exception as e:
                    print(f"Error stopping instance {instance_id}: {str(e)}")
                    doc_ref.update({'currentStep': 'Error stopping instance', 'trainingStatus': 'error'})

            threading.Thread(target=stop_instance, args=(instanceId,)).start()
            return JsonResponse({'error': f'Error during training: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def fix_tokenizer(tokenizer, new_lang):
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = new_lang
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}
    print("Tokenizer fixed")

def get_batch(df, batch_size):
    batch = df.sample(batch_size)
    source_texts = batch.iloc[:, 0].tolist() 
    target_texts = batch.iloc[:, 1].tolist()
    print("Get batch finished") 
    return source_texts, target_texts
