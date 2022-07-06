#!/usr/bin/env python3

# Authors: Linda Moraes, adapted from Tien Tran, who adapted from ROBOTIS 
# Tien Tran mail: quang.tran@fh-dortmund.de

import rospy
import os
import numpy as np
import time
import sys
import csv

from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray

import gym
import math
import gym_turtlebot3
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gym import spaces
from gym.utils import seeding
from std_msgs.msg import Float32MultiArray

EPISODES = 10

import torch as T

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

time.sleep(4)
os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + 1)
rospy.init_node('TurtleBot3_Circuit_Simple-v0'.replace('-', '_') + "_w{}".format(1))
env = gym.make('TurtleBot3_Circuit_Simple-v0', observation_mode=0, continuous=True, env_stage=1)
time.sleep(4)

observation = env.reset(new_random_goals=True, goal=None)

if not os.path.exists('logs70'):
	os.makedirs('logs70')
	
writer = SummaryWriter('logs70')


class LinearDeepQNetwork(nn.Module):
    
    #constructor 
    def __init__(self, lr, n_actions, input_dims):
        # run the constructor of the parent class.
        super(LinearDeepQNetwork,self).__init__()
   
        
        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear (256, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear (256, n_actions)
        #choose gradient decent method for backward propergation
        self.optimizer = optim.Adam(self.parameters(),lr = lr)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cpu')

        self.to(self.device)
        
    def forward(self, state):
        
        layer1 = F.relu (self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        
        # Apply dropout
        drop_out = self.dropout(layer2)
        
        out_actions = self.fc3(drop_out)
        
        return out_actions

class ReinforceAgent():
    def __init__(self, state_size, action_size, writer):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.Path = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.Path.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/torch_model/stage_4_')
        self.resultPATH = self.Path.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/result/result.csv')
        self.result = Float32MultiArray()

        self.load_model = True
  #      self.load_episode = 0
        self.state_size = state_size
        self.observation_space = (self.state_size,)

        self.action_size = action_size
        self.action_space = np.arange(0,5,1) # for similar to gym (action index for scling)
        
        #for loging only, enter the day of run
        self.day = 1905
        
        self.writer = writer
        
        '''Hyperparameters'''
        
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0 # for exploration and exploitation dilema
        
        self.global_step = 0
                 
        self.model = LinearDeepQNetwork(self.learning_rate,  self.action_size, self.observation_space) #action-value function Q
          
        print(self.model)

        #Tien: TODO: Load trained model'
        if self.load_model:
            self.epsilon = 0.3
        #    self.global_step = self.load_episode# dummy for bypass the batch sizes
            print ('Loading model ')
            self.model.load_state_dict(T.load(self.dirPath + '/dqn_stL_model.pth'))
            self.model.eval()
            print ("Load model state dict")
            # TODO: Load previos epsilon self.epsilon = 0.99 
        
    
    def test_goals(self, t, env):
    	if env == 3:
    		if t < 2:
        		return ([0.5, -3.5],[0.5, -3.5])
    		elif 2 <= t < 4:
        		return ([3.5, -0.5],[3.5, -0.5])
    		elif 4 <= t < 6:
        		return ([3.5, -3.5],[3.5, -3.5])
    		elif t >= 6:
        		return ([2.7, -2.0],[2.7, -2.0])
    	else:
    		if t < 5:
        		return ([-1.5, -1.5],[-1.5, -1.5])
    		elif 5 <= t < 10:
        		return ([1.5, -1.5],[1.5, -1.5])
    		elif 10 <= t < 20:
        		return ([-1.5, 1.5],[-1.5, 1.5])
    		elif t >= 20:
        		return ([1.5, 1.5],[1.5, 1.5])

        	
    def choose_action (self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor(observation,dtype= T.float).to(self.model.device) # turn to pytorch tensor           
            actions = self.model(state) # pass to dqn, this is q value, same as self.model.forward(state)
            
            action =T.argmax(actions).item() # Tien: item will take the tensor back to env: interger 
            #print ("Action chose:",  action)
        return action
        
             

if __name__ == '__main__':
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 26
    action_size = 5 #must be odd number

    det = {0: -1.5, 1: -0.75, 2: 0, 3: 0.75, 4: 1.5} # turning actions into angular vel


    agent = ReinforceAgent(state_size, action_size, writer)
    scores, episodes = [], []
    agent.global_step = 0
    start_time = time.time()
    

    for e in tqdm(range(1, EPISODES)):
        done = False
        goal = agent.test_goals(e, 3) # e is the local episode and the number is the stage. Use 3 for L stage and other numbers for others envs (1, 2...)
        state = env.reset(new_random_goals=False, goal=goal)
        score = 0

        for t in range(agent.episode_step):

            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(np.array([det[action], 0.15], dtype=np.float64))
            #learn from sars'tupe
            score += reward
            state = next_state
            
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if t >= 500: #take so long to reach goal: timeout 500 default
                print("Time out!!")
                done = True

            if done:
              
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                
                writer.add_scalar("Reward/train", reward, e)

                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            agent.global_step += 1
