import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'companyId':"COMP37",'jobType':"CEO", 'degree':"MASTERS", 'major':"MATH, 'industry':"HEALTH", 'yearsExperience':10, 'milesFromMetropolis':10})

print(r.json())
