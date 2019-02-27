import json

path = '../../data/marco/train_v2.1.json'
with open(path, 'r') as file:
    data = json.load(file)
print(len(data['query_id']))
print('12')
