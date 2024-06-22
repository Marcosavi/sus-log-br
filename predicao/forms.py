# forms.py
from django import forms

class ForquilhinhaVacinaForm(forms.Form):
    Ano = forms.IntegerField(label='Enter the year you want to compare (2004-2022)')