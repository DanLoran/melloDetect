import requests
resp = requests.post("http://localhost:5000/predict",
                     data="/home/minh/git/melloDetect-deploy/melloDetect/test/test.jpg")
print(resp.json())
