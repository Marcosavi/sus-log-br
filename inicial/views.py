from django.shortcuts import render #type: ignore
from django.views.generic import TemplateView #type: ignore

# Create your views here.
class HomeView(TemplateView):
    template_name = "inicial/index.html"