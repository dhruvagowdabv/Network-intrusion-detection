from django.shortcuts import render

# from django.http import JsonResponse
# from django.http import JsonResponse
# from ML.predict import predict_single

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# live simulation
import random
from ml.log_parser import parse_log_file

from ml.predict import predict_single
                #   for upload html
from django.shortcuts import render
def home(request):
    return render(request, "upload.html")


@csrf_exempt
def predict_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        result = predict_single(data)

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)




                                                    # AI log parser updated

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os
from ml.log_parser import parse_log_file


@csrf_exempt
def upload_log_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    file = request.FILES["file"]

    # Save file temporarily
    file_path = os.path.join("temp_log.csv")

    with open(file_path, "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    try:
        results = parse_log_file(file_path)
        return render(request, "result.html", {"results": results})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)








# for live simulation 
current_index = 0

def live_feed(request):
    global current_index

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(BASE_DIR, "ml", "logs", "demo_logs.csv")

    logs = parse_log_file(file_path)

    if current_index >= len(logs):
        current_index = 0  # loop again

    result = logs[current_index]
    current_index += 1

    return JsonResponse(result)

def live_page(request):
    return render(request, "live.html")













# def health_check(request):
#     return JsonResponse({"status": "Detector API is running"})





# def predict_view(request):
#     # dummy test sample for now
#     sample = {
#         "duration": 0,
#         "protocol_type": "tcp",
#         "service": "http",
#         "flag": "SF",
#         "src_bytes": 200,
#         "dst_bytes": 300,
#         "land": 0,
#         "wrong_fragment": 0,
#         "urgent": 0,
#         "hot": 0,
#         "num_failed_logins": 0,
#         "logged_in": 1,
#         "num_compromised": 0,
#         "root_shell": 0,
#         "su_attempted": 0,
#         "num_root": 0,
#         "num_file_creations": 0,
#         "num_shells": 0,
#         "num_access_files": 0,
#         "num_outbound_cmds": 0,
#         "is_host_login": 0,
#         "is_guest_login": 0,
#         "count": 1,
#         "srv_count": 1,
#         "serror_rate": 0.0,
#         "srv_serror_rate": 0.0,
#         "rerror_rate": 0.0,
#         "srv_rerror_rate": 0.0,
#         "same_srv_rate": 1.0,
#         "diff_srv_rate": 0.0,
#         "srv_diff_host_rate": 0.0,
#         "dst_host_count": 1,
#         "dst_host_srv_count": 1,
#         "dst_host_same_srv_rate": 1.0,
#         "dst_host_diff_srv_rate": 0.0,
#         "dst_host_same_src_port_rate": 1.0,
#         "dst_host_srv_diff_host_rate": 0.0,
#         "dst_host_serror_rate": 0.0,
#         "dst_host_srv_serror_rate": 0.0,
#         "dst_host_rerror_rate": 0.0,
#         "dst_host_srv_rerror_rate": 0.0
#     }

#     prediction, probability = predict_single(sample)

#     return JsonResponse({
#         "prediction": "Attack" if prediction == 1 else "Normal",
#         "confidence": round(float(probability), 4)
#     })