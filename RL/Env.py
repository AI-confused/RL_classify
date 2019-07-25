import json
import numpy as np
import csv


with open('dataset/apptype_id_name.json', 'r') as f:
    apptypes = json.load(f)
    # print(contents)
top_action_space = []
buttom_action_space = []
for _ in range(26):
    top_action_space.append(apptypes[_]['type_id'])
for _ in range(26, len(apptypes)):
    buttom_action_space.append(apptypes[_]['type_id'])
# print(action_space)

reader = csv.reader(open('dataset/train_set.csv', encoding='utf-8'))
train_data_list = []
for i, _ in enumerate(reader):
    if i != 0:
        train_data_list.append(_)
max_lable_count = 4025
with open('dataset/count_train_button_type.json', 'r') as f:
    type_dic = json.load(f)
# with open('apptype_train_vec.json', 'r') as f:
#     contents = json.load(f)
# # print(type(contents))
# for content in contents:
#     content['vector'] = np.array(content['vector'])
# print(contents)
# print(array.shape)

class env_classfy(object):
    def __init__(self):
        self.action_space = buttom_action_space
        self.n_actions = len(self.action_space)
        # self.point = 0
        # self.maxpoint = len(train_data_list)
        # print('maxpoint ' + str(self.maxpoint))
        self.done = 0
        self.right_counter = 0
        # self.ob_sum = 0

    def reset(self):
        # observation = np.array(json.loads(train_data_list[0][0]))
        self.done = 0
        self.minority_miss_count = 0
        self.half_miss_count = 0
    #     # self.point = 0
    #     # self.labels
    #     # self.label = train_data_list[0][1]
    #     # print(self.labels)
    #     # for _ in contents[self.point]['type_id']:
    #     #     self.labels.append(''.join(list(_)[:4]))
    #     # print(self.labels)
    #     # self.point += 1
    #     return observation

    def step(self, action, label):
        # self.point += 1
        # if self.point == self.maxpoint-1:
        #     self.done = 1
            # print(self.point)
        # if not self.done:
        type_minority_flag = 0
        type_majority_flag = 0
        type_half_flag = 0
        # for _ in self.labels:
            # if _:
        if type_dic[label]/max_lable_count <= 0.1:
            type_minority_flag = 1
        elif type_dic[label]/max_lable_count > 0.1 and type_dic[label]/max_lable_count <= 0.5:
            type_half_flag = 1
        else:
            type_majority_flag = 1

        if buttom_action_space[action] == label:
            if type_minority_flag:
                reward = 1
            elif type_half_flag:
                reward = 0.5
            elif type_majority_flag:
                reward = 0.1
        else:
            if type_minority_flag:
                reward = -1
                self.minority_miss_count += 1
                # self.done = 1
            elif type_half_flag:
                reward = -0.5
                self.half_miss_count += 1
            elif type_majority_flag:
                reward = -0.1
           # self.right_counter += 1
        if self.minority_miss_count >= 1000 or self.half_miss_count >= 5000:
            self.done = 1
        # s_ = np.array(json.loads(train_data_list[self.point][0]))
        # self.labels = []
        # self.label = train_data_list[self.point][1]
        # print(self.labels)
        # for _ in contents[self.point]['type_id']:
        #     self.labels.append(''.join(list(_)[:4]))
        # print(self.labels)
        return reward, self.done

