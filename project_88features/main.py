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
# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 8000001
#max_episodes = 1
keep_prob = 1
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        
        #self.cnnlayer1 = torch.nn.Sequential(
        #torch.nn.Conv2d(9,36, kernel_size=2, stride=1, padding=0),
        #torch.nn.ReLU(),
        #torch.nn.MaxPool2d(kernel_size=2),
        #torch.nn.Dropout(p=1 - keep_prob))
        #self.cnnlayer2 = nn.Linear(36, num_actions )
    
    def forward(self, state):
        #flat_state = np.ndarray.flatten(state)
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        #state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)
        #policy_dist = F.relu(self.cnnlayer1(state))
        #policy_dist = policy_dist.view(policy_dist.size(0), -1)   # Flatten them for FC
        #policy_dist = F.softmax(self.cnnlayer2(policy_dist), dim=1)

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
    #num_inputs = env.observation_space.shape[0]
    #num_outputs = env.action_space.n
    num_inputs = 88
    num_outputs = 3
    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    #best_model = torch.load('200000ILnewmodel3.ckpt')
    #actor_critic.load_state_dict(best_model)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    avg_rewards = 0
    
    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

       # finish = False
        
        #state = env.reset()
        bitmaps4x4, bitmaps4x1 =  np.array(env.generate_env(episode))
        flatten_state1 = np.ndarray.flatten(bitmaps4x4)
        flatten_state2 = np.ndarray.flatten(bitmaps4x1)
        flatten_state = np.concatenate((flatten_state1, flatten_state2), axis=0)
        
        #flatten_state = state
        if((episode%3 == 0)):
            actions = env.generatelabel_env(episode)
         #   print(actions)
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(flatten_state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 
            
            #action = np.random.choice(num_outputs, p=[0.3, 0.3, 0.4])
            
           # action = np.random.choice(num_outputs, p=np.squeeze(dist))
            #if(finish == True):
             #   action = action_list.index('finish')
            if(episode%3 == 0):
              #  print(steps)
                action = action_list.index(actions[steps])
            else:
                action = np.random.choice(num_outputs, p=np.squeeze(dist))
            #m = Categorical(policy_dist)
            #action = m.sample()
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            #new_state, reward, done, _ = env.step(action)
            new_bitmaps4x4, new_bitmaps4x1 ,reward, done, finish = env.next_state(bitmaps4x4,bitmaps4x1,action,steps)
            flatten_new_state1 = np.ndarray.flatten(new_bitmaps4x4)
            flatten_new_state2 = np.ndarray.flatten(new_bitmaps4x1)
            flatten_new_state = np.concatenate((flatten_new_state1, flatten_new_state2), axis=0)
            #flatten_new_state = new_state
            
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            bitmaps4x4 = new_bitmaps4x4
            bitmaps4x1 = new_bitmaps4x1
            
            if done or steps == num_steps-1:
                Qval, _ = actor_critic.forward(flatten_new_state)
                Qval = Qval.detach().numpy()[0,0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 4 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break
            
            
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
        if(episode % 100000 == 0):
            torch.save(actor_critic.state_dict(), str( episode) + 'ILrew0model.ckpt')    
        
        avg_rewards = sum(all_rewards) / (episode+1)
        print('average reward', avg_rewards)
    
    # Plot results
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
    num_outputs = 3
    testmodel = ActorCritic(num_inputs, num_outputs, hidden_size)
    best_model = torch.load('200000ILrew0model.ckpt')
    testmodel.load_state_dict(best_model)
    predicted_actions = []
    
    correct = 0
    correct_and_min = 0
    total = 0
    with torch.no_grad():
        for i in range(100000,100399):
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
                #print(policy_dist)
                #print(action_list[predicted])
                predicted_actions.append(action_list[predicted])
                #current_state = new_state
            total += 1
            if(finish == True):
                predicted_actions.append('finish')
            print(labelled_action)
            print(predicted_actions)

            if(labelled_action == predicted_actions):
                correct_and_min +=1
            
            predicted_actions.clear()
           # if(np.array_equiv(current_state[0:len(pregrid)], current_state[5:len(postgrid)])):
           #     correct += 1
            if(np.array_equiv(new_bitmaps4x4[pregrid_pos], np.ones(( 4,  4)))):
                correct += 1
                
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct_and_min / total))
    
if __name__ == "__main__":
    #env = gym.make("CartPole-v0")
    env = GridWorld()
   # a2c(env) 
    test(env)
    
    
    
