import json
import numpy
import random
import gensim
from gensim.models import KeyedVectors
import numpy as np
import jieba
import csv


with open('dataset/apptype_id_name.json', 'r') as f:
    contents0 = json.load(f)
button_types = []
for _ in range(26, len(contents0)):
    button_types.append(contents0[_]['type_id'])
# arr0 = {}
# for _ in button_types:
#     arr0[_] = 0
# # print(arr0)
# with open('dataset/train_set.csv', 'r') as f:
#     contents1 = csv.reader(f)
#     for i, _ in enumerate(contents1):
#         if i != 0:
#             arr0[_[4]] += 1
#             if _[5]:
#                 arr0[_[5]] += 1
#             # if arr0[_[5]] != '':
#             #
#     # for item in _:
#     #     # arr0[''.join(list(item)[:4])] += 1
#     #     arr0[item] += 1
# count = 0
# for _ in button_types:
#     count += arr0[_]
#
# print(count)
with open('dataset/count_train_button_type.json', 'r') as f:
    c = json.load(f)
    max = 0
    for _ in button_types:
        if max < c[_]:
            max = c[_]
print(max)
    # f.write(json.dumps(arr0))
    # with open('dataset/top_types.json', 'w') as f:
    #     f.write(json.dumps(top_types))