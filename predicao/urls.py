from predicao import views
from django.urls import path #type: ignore

app_name = "predicao"

urlpatterns = [
    path("", views.predicao_view, name="predicao"),
]