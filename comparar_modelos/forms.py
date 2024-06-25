from django import forms

class CompararModelsForm(forms.Form):
    YEAR_CHOICES = [(year, year) for year in range(2004, 2023)]
    Ano = forms.ChoiceField(choices=YEAR_CHOICES, label="Ano")
