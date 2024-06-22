# forms.py
from django import forms

class ForquilhinhaVacinaForm(forms.Form):
    Ano = forms.IntegerField(label='Adicione o ano que você deseja comparar (entre 2004 e 2022):')