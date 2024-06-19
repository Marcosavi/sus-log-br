from inicial import views
from django.urls import path #type: ignore

urlpatterns = [
    path("", views.HomeView.as_view(), name = "home"),
]