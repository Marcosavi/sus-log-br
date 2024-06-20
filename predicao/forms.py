from django import forms #type: ignore
from .models import Predicao #type: ignore

class PredicaoForm(forms.ModelForm):
    class Meta:
        model = Predicao
        fields = '__all__'