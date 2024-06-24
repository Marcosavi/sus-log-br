from predicao import views
from django.urls import path #type: ignore

app_name = "predicao"

urlpatterns = [
    path("passado/", views.compare_year, name="passado"),
    path("futuro/", views.future_predictions, name="futuro"),
]