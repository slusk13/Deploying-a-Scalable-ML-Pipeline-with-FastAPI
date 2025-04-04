import json
import requests

# Send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

# Print Status Code
print(f"GET Status Code: {r.status_code}")
# Print Welcome Message
print(f"GET Response: {r.json()['message']}")


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST using the data above
r = requests.post(
    "http://127.0.0.1:8000/data/",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

# Print the status code
print(f"POST Status Code: {r.status_code}")
# Print the result
print(f"POST Response: {r.json()['result']}")
