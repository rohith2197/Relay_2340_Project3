from django.urls import path
from . import views

app_name = "chat"
urlpatterns = [
    path("", views.index, name="index"),
    path("api/chat/", views.ask, name="ask"),
    path("health/", views.health, name="health"),
]
