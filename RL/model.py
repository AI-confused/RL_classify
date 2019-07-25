import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import json


#hyper parameters
with open('dataset/apptype_id_name.json', 'r') as f:
    contents = json.load(f)
action_space = []
for _ in range(26, len(contents)):
    action_space.append(contents[_]['type_id'])
# print(action_space)
batch_size = 64
lr = 0.00025
epsilon = 0.9
gamma = 0.1
target_replace_iter = 3000
n_action = len(action_space)
# print(n_action)
# print(len(contents))
n_state = 300

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.action_space = action_space

        self.linear1 = nn.Linear(n_state, 250)
        self.linear1.weight.data.normal_(0.001, 0.1)

        self.linear2 = nn.Linear(250, 250)
        self.linear2.weight.data.normal_(0.001, 0.1)

        # self.linear3 = nn.Linear(256, 128)
        # self.linear3.weight.data.normal_(0, 0.1)


        self.out = nn.Linear(250, n_action)
        self.out.weight.data.normal_(0.001, 0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        # x = self.linear3(x)
        # x = F.relu(x)
        x = self.out(x)
        #x = F.softmax(x, 1)
        return x

class DQN(object):
    def __init__(self):
        # self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.eval_net, self.target_net = Network().cuda(), Network().cuda()
        # print(self.eval_net, self.target_net)
        self.memory_capacity = 30000
        self.memory_conter = 0
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_capacity, n_state * 2 + 3))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r], s_, done))

        # replace the old memory with new memory
        index = self.memory_conter % self.memory_capacity
        self.memory[index, :] = transition

        self.memory_conter += 1

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0)).cuda()
        # action_value_prob = self.eval_net.forward(x)
        # action = np.random.choice(range(action_value_prob.shape[1]), p=action_value_prob.detach().numpy().ravel())
        # print(action)
        if np.random.uniform() < epsilon:
            action_value = self.eval_net.forward(x)
            action_value = action_value.cpu().detach().numpy()
        #     # print(action_value)
        #     # print(np.argmax(action_value, 1))
            action = np.argmax(action_value, 1)[0]
        #     # action = action_space[action]
        #     # print(action)
        else:
            action = np.random.randint(0, n_action)
            # action = action_space[action]
            # print(action)
        return action

    def learn(self):
        #target net update
        if self.learn_step_counter % target_replace_iter == 0:
            # print(self.learn_step_counter)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # for i in self.target_net.named_parameters():
            #     print(i)

        sample_index = np.random.choice(self.memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        # print(b_memory)
        b_s = Variable(torch.Tensor(b_memory[:, :n_state])).cuda()
        # print(b_s.shape)
        b_a = Variable(torch.LongTensor(b_memory[:, n_state:n_state+1].astype(int))).cuda()
        # print(b_a)
        b_r = Variable(torch.FloatTensor(b_memory[:, n_state+1:n_state+2])).cuda()
        # print(b_r)
        b_s_ = Variable(torch.FloatTensor(b_memory[:, n_state+2:-1])).cuda()
        # print(b_s_.shape)
        b_done = Variable(torch.FloatTensor(b_memory[:, -1:])).cuda()
        # print(b_done)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        # print(self.eval_net(b_s))
        # print(q_eval)
        q_next = self.target_net(b_s_).detach()#no backward
        # print(q_next)
        q_target = b_r + gamma*q_next.max(1)[0].unsqueeze(1)  #choose max action value
        self.loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
