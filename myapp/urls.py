from django.urls import path
from .views import *

urlpatterns = [
    path('translate/', translation_view, name='translation'),
    path('get-details/', get_details, name='get_method'),
]
