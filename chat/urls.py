from django.urls import path
from . import views
from . import admin_views
from . import knowledge_views

app_name = "chat"
urlpatterns = [
    path("", views.index, name="index"),
    path("api/chat/", views.ask, name="ask"),
    path("health/", views.health, name="health"),
    path("dashboard/performance/", admin_views.performance, name="admin_performance"),
    path("dashboard/logs/", admin_views.logs, name="admin_logs"),
    path("dashboard/knowledge/", knowledge_views.knowledge_base, name="admin_knowledge"),
]
