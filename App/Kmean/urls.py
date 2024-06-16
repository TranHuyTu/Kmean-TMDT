from django.urls import path
from . import views

urlpatterns = [
    path("kmean/", views.kmean, name="Kmean"),
    path("findCountAge/", views.findCountAge, name="findCountAge"),
]
