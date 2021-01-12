from django.urls import path
from .views import Handwriting


urlpatterns = [
    path('', Handwriting.as_view(), name='handwriting_sample'),
]
