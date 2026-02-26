import requests
import json
import os

json_file = "doctors.json"

def fetch_and_save_doctors():
  url = "https://mysiloam-api.siloamhospitals.com/api/v2/doctors/ai/withavailability?show=100&page=1"
  payload = {}
  headers = {
    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7InVzZXJJRCI6IjU5MDM2NCIsInBob25lTnVtYmVyIjoiKzYyODIyNjE3MDA2NjIiLCJwYXRpZW50SWQiOiIwNDI2MTcwNC03MzM2LWNkNDEtNjQxOS03NzBjMTEyMGM3ZTYifSwiaWF0IjoxNzQ4NDE2Mzg0LCJleHAiOjQ3NzMxNjgwMDB9.PUxGwwGN2M-ZW2Ys3lWjU3uhib51h8g1LDKkqTlgum4'
  }
  response = requests.request("GET", url, headers=headers, data=payload)
  data = response.json()
  with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  print(f"Data saved to {json_file}")

def load_doctors_from_file():
  with open(json_file, "r", encoding="utf-8") as f:
    return json.load(f)

if not os.path.exists(json_file):
  fetch_and_save_doctors()
else:
  print(f"Loading data from {json_file}")
doctors_data = load_doctors_from_file()
print(doctors_data)