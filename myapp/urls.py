from django.urls import path
from .views import translation_view

urlpatterns = [
    path('translate/', translation_view, name='translation'),
]
