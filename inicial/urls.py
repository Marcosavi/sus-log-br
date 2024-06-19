from . import views
from django.urls import path #type: ignore

app_name = "inicial"

urlpatterns = [
    path("", views.HomeView.as_view(), name="index"),
]