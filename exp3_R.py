import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
from random import shuffle
np.seterr(divide='ignore', invalid='ignore')


class exp3R_bandit: 
    
    def __init__(self, num_actions, init_weights=None, gamma=0.1, H=1000, delta=0.1):
        assert(num_actions > 1)        
        assert(gamma > 0 and gamma < 1)
        if init_weights is not None: 
            assert(num_actions == len(init_weights))
            self.init_weights = init_weights
        else: 
            self.init_weights = [1.0] * self.num_actions 
        self.weights = None
        self.H = H 
        self.delta = delta 
        self.t = 0 
        self.total_rewards = 0 
        self.num_actions = num_actions 
        self.prob_dist = tuple([1.0/self.num_actions] * self.num_actions)
        self.gamma = gamma
        self.reset_exp3_weights()
        self.reset_interval() 
        self.I = 1 
        self.e = math.sqrt ((self.num_actions * math.log(1/self.delta)) / (2 * self.gamma * self.H))

        
    def reset_interval(self): 
        self.uniform_draws = dict.fromkeys(range(self.num_actions), 0)         
        self.rewards_in_interval = dict.fromkeys(range(self.num_actions), 0) 

        
    def reset_exp3_weights(self): 
        if self.weights is not None:   
            self.weights = self.init_weights   
            self.prob_dist = tuple([1.0/self.num_actions] * self.num_actions)
        
        
    def detect_drift(self):           
        actions_weights = list(zip(range(self.num_actions), self.weights)) 
        actions_weights = sorted(actions_weights, key=lambda tup: tup[1], reverse=True)
        kmax, _ = actions_weights[2]
        mu_kmax = self.rewards_in_interval[kmax] / self.uniform_draws[kmax]
        challengers =  [i[0] for i in actions_weights[3:]]
        for k in challengers: 
            mu_k = self.rewards_in_interval[k] / self.uniform_draws[k]
            print (mu_k - mu_kmax, 2 * self.e)
            if mu_k - mu_kmax >= 2 * self.e: 
                return True     
        return False 
    
    
    def check_drift(self): 
        min_uniform_draws = min(self.uniform_draws.values())
        if min_uniform_draws >= (self.gamma * self.H) / (self.num_actions): 
            if self.detect_drift():
                self.reset_exp3_weights() 
                self.reset_interval() 
            else: 
                self.reset_interval() 
            self.I += 1 
        return False 

    
    def get_prob_dist(self): 
        return self.prob_dist
    
    
    def update_prob_dist(self): 
        the_sum = float(sum(self.weights))
        self.prob_dist = tuple((1.0 - self.gamma) * (w / the_sum) + (self.gamma / len(self.weights)) for w in self.weights)
        # self.prob_dist = tuple(normalize(self.prob_dist))
    
    def get_display_options(self): 
        bandit_policy_options = np.zeros(self.num_actions)
        attempts = 0 
        while np.sum(bandit_policy_options) < 3:
            attempts += 1 
            if attempts > 100: 
                # break infinite loop. 
                # Needs to be fixed
                break
            try: 
                choice = np.random.choice(np.arange(0,self.num_actions), p=self.prob_dist)    
                bandit_policy_options[choice] = 1
                self.uniform_draws[choice] += 1
            except ValueError: 
                print (self.prob_dist)
                print (self.weights)
                print (sum(self.prob_dist))
        return bandit_policy_options         
    
    
    def update_weights(self, displayed_options, rewards): 
        assert(len(displayed_options) == len(rewards))
        self.t += 1         
        displayed_options = displayed_options.astype(np.int32) 
        rewards = rewards.astype(np.int32)
        for index, (display, reward) in enumerate(list(zip(displayed_options, rewards))):
            if display == 1: 
                self.rewards_in_interval[index] += reward
                self.total_rewards += reward
                estimated_reward = 1.0 * reward / self.prob_dist[index]
                self.weights[index] *= math.exp(estimated_reward * self.gamma / self.num_actions) 
        self.update_prob_dist()       
        self.weights = normalize(self.weights)
        self.check_drift() 
                
    
    def get_weights(self): 
        return self.weights 
    
    
    def get_total_rewards(self): 
        return self.total_rewards
    
