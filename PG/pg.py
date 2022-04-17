import gym
import math
import torch
import numpy as np
import torch.nn as nn
from multiprocessing import Pool, Process


class pg():

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = nn.Sequential(
                nn.Conv2d(3, 10, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
                nn.Conv2d(10, 10, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
                nn.Conv2d(10, 10, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
                nn.Conv2d(10, 1, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
                nn.Flatten(),
                nn.Linear(36, 6)
                )
        self.policy.to(self.device)
        self.dataset = []
        self.buffer = 100

    def save(self):
        torch.save(self.policy.state_dict(), 'net.pth')

    def load(self):
        self.policy.load_state_dict(torch.load('net.pth', map_location='cpu'))
        self.policy.to(self.device)

    def forget(self):
        while (len(self.dataset) > self.buffer):
            self.dataset.pop(0)

    def reset(self):
        self.dataset = []

    def explore(self):

        env = gym.make('CarRacing-v1')
        state = env.reset()
    

        done = False
        policy = self.policy.cpu().eval()

        state_pool = []
        action_pool = []
        reward_pool = []

        with torch.no_grad():
            while not done:

                state = np.transpose(state, (2, 0, 1))
                state_pool.append(state)
                state = torch.from_numpy(state).float()

                action_params = policy(state)[0]
                action_params[1] = torch.abs(action_params[1])
                action_params[3] = torch.abs(action_params[3])
                action_params[5] = torch.abs(action_params[5])
                normal_1 = torch.distributions.normal.Normal(action_params[0], action_params[1])
                normal_2 = torch.distributions.normal.Normal(action_params[2], action_params[3])
                normal_3 = torch.distributions.normal.Normal(action_params[4], action_params[5])
                action_1 = torch.sigmoid(normal_1.sample()).item()
                action_2 = torch.tanh(normal_2.sample()).item()
                action_3 = torch.tanh(normal_3.sample()).item()
                action = np.array([action_1, action_2, action_3])
                action_pool.append(action)

                next_state, reward, done, _ = env.step(action)
                reward_pool.append(reward)
                state = next_state

        return self.dataset.append([state_pool, action_pool, reward_pool])

    def pool_explore(self, num_workers, num_loops):
        # pool = Pool(num_workers)
        # data = pool.starmap(self.explore, [() for _ in range(num_loops)])
        # pool.close()
        # pool.join()
        pass

    def train(self):

        NUM = 0
        LR = 0.001

        self.policy.train()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)

        for trajectory in self.dataset:

            state = np.array(trajectory[0])
            action = np.array(trajectory[1])
            reward = np.array(trajectory[2])

            state = torch.from_numpy(state).to(self.device).float()
            action = torch.from_numpy(action).to(self.device).float()
            reward = torch.tensor(sum(reward)).to(self.device).float()

            action_params = self.policy(state)
            action_params_sigma_1 = torch.abs(action_params[:,1])
            action_params_sigma_2 = torch.abs(action_params[:,3])
            action_params_sigma_3 = torch.abs(action_params[:,5])
            normal_1 = torch.distributions.normal.Normal(action_params[:,0], action_params_sigma_1)
            normal_2 = torch.distributions.normal.Normal(action_params[:,2], action_params_sigma_2)
            normal_3 = torch.distributions.normal.Normal(action_params[:,4], action_params_sigma_3)
            log_prob_1 = normal_1.log_prob(action[:,0])
            log_prob_2 = normal_2.log_prob(action[:,1])
            log_prob_3 = normal_3.log_prob(action[:,2])
            
            log_prob_1 = torch.sum(log_prob_1)
            log_prob_2 = torch.sum(log_prob_2)
            log_prob_3 = torch.sum(log_prob_3)

            log_prob = log_prob_1 + log_prob_2 + log_prob_3
            loss = -log_prob * reward
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            NUM = NUM + 1
            print('Training')
            print('Loop = {}/{}; Loss = {}'.format(NUM, len(self.dataset), loss.detach().cpu().item()))

    def valid(self):
        env = gym.make('CarRacing-v1')
        state = env.reset()

        done = False
        policy = self.policy.cpu().eval()

        reward_pool = []

        with torch.no_grad():
            while not done:
                # env.render()

                state = np.transpose(state, (2, 0, 1))
                state = torch.from_numpy(state).float()

                action_params = policy(state)[0]
                action_params[1] = torch.abs(action_params[1])
                action_params[3] = torch.abs(action_params[3])
                action_params[5] = torch.abs(action_params[5])
                normal_1 = torch.distributions.normal.Normal(action_params[0], action_params[1])
                normal_2 = torch.distributions.normal.Normal(action_params[2], action_params[3])
                normal_3 = torch.distributions.normal.Normal(action_params[4], action_params[5])
                action_1 = torch.sigmoid(normal_1.sample()).item()
                action_2 = torch.tanh(normal_2.sample()).item()
                action_3 = torch.tanh(normal_3.sample()).item()
                action = np.array([action_1, action_2, action_3])
                next_state, reward, done, _ = env.step(action)

                reward_pool.append(reward)
                state = next_state

        str_to_write = str(sum(reward_pool)) + '\n'
        print('reward = ' + str_to_write)
        file = open('reward.txt', 'a')
        file.write(str_to_write)
        file.close()


if __name__ == "__main__":

    demo = pg()
    for i in range(1000):
        print('********************')
        print("Epoch = " + str(i))
        print('********************')
        for i in range(8):
            demo.explore()
        demo.forget()
        demo.train()
        demo.valid()

    print('Finish!')

