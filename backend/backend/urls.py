"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# from detector.views import health_check
from detector.views import predict_view

# for AI log parser
from detector.views import upload_log_view
# for result ui
from detector.views import home

# for live simulation
from detector.views import live_feed
from detector.views import live_page

urlpatterns = [
    path('admin/', admin.site.urls),
    # for result ui
    path('', home),
    path('predict/', predict_view),
    path('upload-log/', upload_log_view),
    # live simulation
    path('live/', live_feed),
    path('live-ui/', live_page),

]
