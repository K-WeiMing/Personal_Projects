import requests

URL = "http://127.0.0.1:8000/predict/image"
FILE_PATH = "../data/images/maksssksksss100.png"

file = {"file": open(FILE_PATH, "rb")}
print(file)
response = requests.post(URL, files=file)

print(response.status_code, response.json())
