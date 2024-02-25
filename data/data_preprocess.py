import json
import random

# generate natural_text for c4_data
# texts = []
# for i in range(10):
#     fr = open(f"c4_{i}.json")
#     json_file = json.load(fr)
#     for row in json_file["rows"]:
#         texts.append(row["row"]["text"])

# dataset = []
# for text in texts:
#     words = text.split(" ")
#     if len(words)>300:
#         dataset.append({"input":" ".join(words[:50]),"output":" ".join(words[50:])})

# fw = open("c4_ref.json", "w")
# json.dump(dataset, fw, indent=4)


# # generate natural_text for alpaca_data
# fr = open("full_alpaca_data.json")
# fw = open("alpaca_ref.json", "w")
# texts = json.load(fr)
# dataset = []
# for text in texts:
#     if len(text['output'].split(" "))>150:
#         dataset.append(text)
        
# print(len(dataset))        
# json.dump(dataset[322:822], fw, indent=4)

fr = open("com_diverse_data_u.json")
dataset=json.load(fr)
result_dict=[]
for data in dataset[:20]:
    for _ in range(50):
        result_dict.append({'input':data['input'], 'output':""})
    
fw = open("com_diverse_data.json", 'w')
json.dump(result_dict, fw, indent=4)

fr = open("chat_diverse_data_u.json")
dataset=json.load(fr)
result_dict=[]
for data in dataset[:20]:
    for _ in range(50):
        result_dict.append({'instruction':data['instruction'], 'input':"", 'output':""})
    
fw = open("chat_diverse_data.json", 'w')
json.dump(result_dict, fw, indent=4)