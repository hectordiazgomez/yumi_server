from django.http import JsonResponse
from django.views.decorators.http import require_GET
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

MODEL_URL = "hectordiazgomez/nllb-spa-awa-v3"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_URL)
tokenizer = NllbTokenizer.from_pretrained(MODEL_URL)

@require_GET
def get_details(request):
    if request.method == "GET":
        return JsonResponse({"Hola": "Api funcionando"})

def fix_tokenizer(tokenizer, new_lang='agr_Latn'):
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

fix_tokenizer(tokenizer)

def translate(text, model, tokenizer, src_lang, tgt_lang, max_length='auto', num_beams=4, n_out=None, **kwargs):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if max_length == 'auto':
        max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
    model.eval()
    generated_tokens = model.generate(
        **encoded.to(model.device),
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=n_out or 1,
        **kwargs
    )
    out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return out[0] if isinstance(text, str) and n_out is None else out

@require_GET
def translation_view(request):
    src_lang = request.GET.get('source_language', 'agr_Latn')
    tgt_lang = request.GET.get('target_language', 'spa_Latn')
    text = request.GET.get('text')

    if not text:
        return JsonResponse({'error': 'Missing text parameter'}, status=400)

    try:
        translation = translate(text, model, tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
        return JsonResponse({'translation': translation})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
