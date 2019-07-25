from classfy_env import env_classfy
from model import DQN
import json
import numpy as np
from torch.autograd import Variable
import torch
import csv


test_data_list = []
with open('dataset/test_set.csv', 'r') as f:
    reader = csv.reader(f)
    for i, _ in enumerate(reader):
        if i != 0:
            test_data_list.append(_)

def run_classfy():
    step = 0
    for episode in range(10000):
       # print('episode: %d' % episode)
        # print('episode ' + str(episode))
        observation = env.reset()
        # print('reset observation' + str(observation))
        # print(observation.shape)
        # print(type(observation))
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # print('action ' + str(action))
            # print(action)
            # print(type(action))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # print('step')
            # print(reward, done)
            # if done:
            #     print('********************8')
            #     print(step)
            RL.store_transition(observation, action, reward, observation_)
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
            observation = observation_

            # break while loop when end of this episode
            step += 1
            if done:
                for point in range(len(test_data_list)):
                    s = np.array(json.loads(test_data_list[point][1]))
                    # labels = []
                    label = test_data_list[point][4]
                    # for _ in sentences[point]['type_id']:
                    #     labels.append(''.join(list(_)[:4]))
                    #action = RL.choose_action(s)
                    s = Variable(torch.unsqueeze(torch.FloatTensor(s), 0))
                    action_value = RL.eval_net(s).detach()
                    action_value = action_value.numpy()
                    # print(action_value)
                    # print(np.argmax(action_value, 1))
                    action = np.argmax(action_value, 1)[0]
                    #action = action_space[action]
                    if action_space[action] == label:
                        env.right_counter += 1
                accuracy = env.right_counter / len(test_data_list)
                print(env.right_counter)
                print('episode: %d' % episode, '| accuracy: %f' % accuracy)
                env.right_counter = 0
                # print(done)
                # print(step)
                break


    print('classfy over')



if __name__ == '__main__':
    with open('dataset/apptype_id_name.json', 'r') as f:
        contents0 = json.load(f)
        # print(contents)
    action_space = []
    for _ in range(26, len(contents0)):
        action_space.append(contents0[_]['type_id'])
    env = env_classfy()
    RL = DQN()
    run_classfy()
