import requests

url = 'http://localhost:5000/predict'
laptop_sample = {
    'manufacturer': 'Dell',
    'category': 'Ultrabook',
    'screen_size': 12.5,
    'screen': 'Full HD 1920x1080',
    'cpu': 'Intel Core i5 6300U 2.4GHz',
    'ram': 8.0,
    'storage': '256GB',
    'gpu': 'Intel HD Graphics 620',
    'operating_system': 'Windows',
    'operating_system_version': 7,
    'weight': 1.26,
}
response = requests.post(url, json=laptop_sample)
print(response.json())