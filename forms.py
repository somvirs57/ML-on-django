from django import forms
from .models import *

class HandwritingForm(forms.ModelForm):
    class Meta:
        model = HandwritingModel
        fields = ('image',)
