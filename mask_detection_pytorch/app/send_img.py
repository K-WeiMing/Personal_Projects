import requests
import pathlib

# URL = "http://127.0.0.1:8000/predict/image" # Localhost
URL = "http://127.0.0.1:8080/predict/image"  # When running Docker


CURR_DIR = pathlib.Path(__file__).resolve().parent
FILE_PATH = "../data/images/maksssksksss100.png"
FILE_PATH = pathlib.Path(CURR_DIR, FILE_PATH)

file = {"file": open(FILE_PATH, "rb")}
print(file)
response = requests.post(URL, files=file)

print(response.status_code, response.json())
