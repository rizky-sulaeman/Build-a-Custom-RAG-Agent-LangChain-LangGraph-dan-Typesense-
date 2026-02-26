import requests
import json
import os

json_file = "doctors.json"

def fetch_and_save_doctors():
  base_url = "https://mysiloam-api.siloamhospitals.com/api/v2/doctors/ai/withavailability"
  payload = {}
  headers = {
    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7InVzZXJJRCI6IjU5MDM2NCIsInBob25lTnVtYmVyIjoiKzYyODIyNjE3MDA2NjIiLCJwYXRpZW50SWQiOiIwNDI2MTcwNC03MzM2LWNkNDEtNjQxOS03NzBjMTEyMGM3ZTYifSwiaWF0IjoxNzQ4NDE2Mzg0LCJleHAiOjQ3NzMxNjgwMDB9.PUxGwwGN2M-ZW2Ys3lWjU3uhib51h8g1LDKkqTlgum4'
  }
  all_doctors = []
  page = 1
  show = 100
  while True:
    url = f"{base_url}?show={show}&page={page}"
    response = requests.request("GET", url, headers=headers, data=payload)
    data = response.json()
    # If the API returns a dict with a 'data' key, adjust here
    doctors = data.get('data') if isinstance(data, dict) and 'data' in data else data
    if not doctors or (isinstance(doctors, list) and len(doctors) == 0):
      break
    if isinstance(doctors, list):
      all_doctors.extend(doctors)
    else:
      all_doctors.append(doctors)
    print(f"Fetched page {page}, got {len(doctors) if isinstance(doctors, list) else 1} doctors.")
    page += 1
  with open(json_file, "w", encoding="utf-8") as f:
    json.dump(all_doctors, f, ensure_ascii=False, indent=2)
  print(f"Total doctors saved to {json_file}: {len(all_doctors)}")

def load_doctors_from_file():
  with open(json_file, "r", encoding="utf-8") as f:
    return json.load(f)

if not os.path.exists(json_file):
  fetch_and_save_doctors()
else:
  print(f"Loading data from {json_file}")
doctors_data = load_doctors_from_file()
print(doctors_data)