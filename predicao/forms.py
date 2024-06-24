# forms.py
from django import forms

class ForquilhinhaVacinaForm(forms.Form):
    Ano = forms.IntegerField(label='Adicione o ano que você deseja comparar (entre 2004 e 2022):')

class FuturePredictionForm(forms.Form):
    year = forms.IntegerField(label="Adicione o ano que você deseja prever (entre 2023 e 2100):", min_value=2023, max_value=2100)
