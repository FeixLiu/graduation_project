import json

path = '../../data/record/train.json'
with open(path, 'r') as file:
    data = json.load(file)
total = len(data['data'])
total_qas = 0
for i in range(len(data['data'])):
    total_qas += len(data['data'][i]['qas'])
    #print(data['data'][i]['passage']['text'])
    #for j in range(len(data['data'][i]['qas'])):
        #print(data['data'][i]['qas'][j]['query'])
print(float(total_qas) / total)
