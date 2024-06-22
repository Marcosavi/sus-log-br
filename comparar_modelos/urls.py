from comparar_modelos import views
from django.urls import path #type: ignore

app_name = "comparar_modelos"

urlpatterns = [
    path("", views.compare_models_df1, name="comparar_models_df1"),
]