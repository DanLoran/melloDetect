import requests
from PIL import Image
resp = requests.post("http://localhost:5000/predict",
                     data=open("/Users/minhtruong/Documents/git/melloDetect-deploy/melloDetect/test/test.jpg", 'rb'))
print(resp.json())
