from classfy_env import env_classfy
from model import DQN
import json
import numpy as np
from torch.autograd import Variable
import torch
import csv

with open('dataset/apptype_id_name.json', 'r') as f:
    contents0 = json.load(f)
    # print(contents)
action_space = []
for _ in range(26, len(contents0)):
    action_space.append(contents0[_]['type_id'])

# test_data_list = []
with open('dataset/train_tuple.json', 'r') as f:
    train_data_list = json.load(f)
# print(train_data_list)
# print(test_data_list)
with open('dataset/test_tuple.json', 'r') as f:
    test_data_list = json.load(f)

def run_classfy():
    step = 0
    for episode in range(100000):
        env.reset()
        observation = np.array(train_data_list[0][0])
        for _ in range(len(train_data_list)):
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # print('action ' + str(action))
            # print(action)
            # print(type(action))
            # RL take action and get next observation and reward
            reward, done = env.step(action, train_data_list[_][1])
            # print(reward, done)
            if _ == len(train_data_list)-1:
                observation_ = np.zeros(300)
                done = 1
            else:
                observation_ = np.array(train_data_list[_+1][0])
            if done:
                observation_ = np.zeros(300)
            # print('step')
            # print(reward, done)
            # if done:
            #     print('********************8')
            #     print(step)
            RL.store_transition(observation, action, reward, observation_, done)
            # print('Rl memory')
            # print(RL.memory)

            if (RL.memory_conter > RL.memory_capacity) and (RL.memory_conter % 5 == 0):
                # print(RL.memory_conter)
                # pass
                # print('step'+str(step))
                # print('learning...')
                RL.learn()
            # if done:
            #     print(step)
            #     print(env.right_counter)


            # swap observation

            # print(observation)

            # break while loop when end of this episode
            step += 1
            if done:
                break
            observation = observation_
        for point in range(len(test_data_list)):
            s = np.array(test_data_list[point][0])
            # print(s)
            # labels = []
            label = test_data_list[point][1]
            # print(label)
            # for _ in sentences[point]['type_id']:
            #     labels.append(''.join(list(_)[:4]))
            # action = RL.choose_action(s)
            s = Variable(torch.unsqueeze(torch.FloatTensor(s), 0)).cuda()
            action_value = RL.eval_net(s).detach()
            action_value = action_value.cpu().numpy()
            # print(action_value)
            # print(np.argmax(action_value, 1))
            action = np.argmax(action_value, 1)[0]
            # action = action_space[action]
            if action_space[action] == label:
                env.right_counter += 1
        accuracy = env.right_counter / len(test_data_list)
        print(env.right_counter)
        print('episode: %d' % episode, '| accuracy: %f' % accuracy)
        env.right_counter = 0


    print('classfy over')
    torch.save(DQN.eval_net, 'model/net.pkl')



if __name__ == '__main__':
    env = env_classfy()
    RL = DQN()
    run_classfy()
