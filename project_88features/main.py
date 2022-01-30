import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from selfparser import *
from torch.distributions import Categorical
import random 
from warmup_scheduler import GradualWarmupScheduler
# hyperparameters
hidden_size = 128
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 100
max_episodes = 24000
drop_prob = 0

model_mode = str(sys.argv[1])

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        
        self.affine1 = nn.Linear(88, 128)
        
        # actor's layer
        self.action_head = nn.Linear(128, 5)
        
        # critic's layer
        self.value_head = nn.Linear(128, 1)

        self.dropout = nn.Dropout(p=drop_prob)

    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
       
        state = F.relu(self.affine1(state))

        value = self.value_head(state)
        
        policy_dist = F.softmax(self.action_head(state), dim=1)

        return value, policy_dist

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()
    
def a2c(env):
    num_inputs = 88
    num_outputs = 5
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    best_model = torch.load('23999secondrun8.ckpt')
    actor_critic.load_state_dict(best_model)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    
    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    avg_rewards = 0
    acc_rew_step = 0
    iter_num = 0
    lost_epi = 0
    lost_epi_list = []
    for episode in range( max_episodes):
        while(1) :
            print('episode:', episode)
            iter_num += 1
            log_probs = []
            values = []
            rewards = []
            bitmaps4x4, bitmaps4x1 =  np.array(env.generate_env(episode))
            flatten_state1 = np.ndarray.flatten(bitmaps4x4)
            flatten_state2 = np.ndarray.flatten(bitmaps4x1)
            flatten_state = np.concatenate((flatten_state1, flatten_state2), axis=0)
            
            rand = random.sample(range(1, 4), 1)
            if ((rand == [1])):
                actions = env.generatelabel_env(episode)
            for steps in range(num_steps):
                value, policy_dist = actor_critic.forward(flatten_state)
                value = value.detach().numpy()[0,0]
                dist = policy_dist.detach().numpy() 
                _, predicted = torch.max(policy_dist, 1)
                
                if((rand == [1])):
                    action = action_list.index(actions[steps])
       
                else :
                    action = np.random.choice(num_outputs, p=np.squeeze(dist))
            
                
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_bitmaps4x4, new_bitmaps4x1 ,reward, done, finish = env.next_state(bitmaps4x4,bitmaps4x1,action,steps)
                flatten_new_state1 = np.ndarray.flatten(new_bitmaps4x4)
                flatten_new_state2 = np.ndarray.flatten(new_bitmaps4x1)
                flatten_state = np.concatenate((flatten_new_state1, flatten_new_state2), axis=0)
                
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                bitmaps4x4 = new_bitmaps4x4
                bitmaps4x1 = new_bitmaps4x1
                
                if done or steps == num_steps-1:
                    Qval, _ = actor_critic.forward(flatten_state)
                    Qval = Qval.detach().numpy()[0,0]
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(steps)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    if episode % 3 == 0:                    
                        sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                    break
                
                
            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval
            
            #update actor critic
            valuesTensor = torch.FloatTensor(values)
            values.clear()
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)
            
            advantage = Qvals - valuesTensor
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
            
            ac_optimizer.zero_grad()
            ac_loss.backward()
            ac_optimizer.step()
            
            if((episode % 1000 == 0) or (episode == (max_episodes-1))):
                torch.save(actor_critic.state_dict(), str( episode) + 'finalrun1.ckpt')    
            acc_rew_step +=1
            if(acc_rew_step == 100):
                avg_rewards = sum(all_rewards) / (acc_rew_step)
                all_rewards.clear()
                acc_rew_step = 0
            if (iter_num>500):
                lost_epi = lost_epi+1
                lost_epi_list.append(episode)
            if(avg_rewards>0.95 or (iter_num>500) ):
                #avg_rewards = 0
                iter_num = 0
                break
    # Plot results
    print("Lost Episodes Count:", lost_epi)
    print("Lost Episodes :", lost_epi_list)
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()
    
    
def test(env):
    num_inputs = 88
    num_outputs = 5
    testmodel = ActorCritic(num_inputs, num_outputs, hidden_size)
    best_model = torch.load('23000finalrun1.ckpt')
    testmodel.load_state_dict(best_model)
    predicted_actions = []
    
    correct = 0
    correct_and_min = 0
    total = 0
    with torch.no_grad():
        for i in range(100000,102399):
            bitmaps4x4, bitmaps4x1 = env.generate_val(i)
            flatten_state1 = np.ndarray.flatten(bitmaps4x4)
            flatten_state2 = np.ndarray.flatten(bitmaps4x1)
            current_state = np.concatenate((flatten_state1, flatten_state2), axis=0)
            labelled_action = env.generatelabel_val(i)
            done = False
            steps = 0
            while((done is False) and (steps<100)):        
                steps += 1
                value, policy_dist = testmodel(current_state)
                _, predicted = torch.max(policy_dist, 1)
                new_bitmaps4x4, new_bitmaps4x1, reward, done, finish = env.next_state(bitmaps4x4,bitmaps4x1, predicted,steps)
                flatten_state1 = np.ndarray.flatten(new_bitmaps4x4)
                flatten_state2 = np.ndarray.flatten(new_bitmaps4x1)
                current_state = np.concatenate((flatten_state1, flatten_state2), axis=0)
                bitmaps4x4 = new_bitmaps4x4
                bitmaps4x1 = new_bitmaps4x1
                predicted_actions.append(action_list[predicted])
            total += 1
            if(finish == True):
                predicted_actions.append('finish')

            if((np.array_equiv(bitmaps4x4[pregrid_pos], bitmaps4x4[postgrid_pos])) and
        (np.array_equiv(bitmaps4x4[pregrid_mark], bitmaps4x4[postgrid_mark])) and
         (np.array_equiv(bitmaps4x1[pregrid_ori], bitmaps4x1[postgrid_ori]))):
                correct += 1
                if(len(labelled_action) >= len(predicted_actions)):
                    correct_and_min +=1
                    
            predicted_actions.clear()
                
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct_and_min / total))
    
if __name__ == "__main__":
    env = GridWorld()
    if( model_mode == 'train'):
        a2c(env) 
    elif( model_mode == 'test'):
        test(env)
    
    
    
