from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index(resquest):
    return HttpResponse("hello there this the index")
