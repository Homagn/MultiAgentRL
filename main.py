import argparse
import warnings
warnings.filterwarnings("ignore")
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import math
import pygame 
from tensorboard_logger import configure, log_value
import copy

#os.environ["CUDA_VISIBLE_DEVICES"] = 'cpu'

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=80, help="maximum episode length") #60
    parser.add_argument("--num-episodes", type=int, default=120000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp") #32
    parser.add_argument("--update-every", type=int, default=100, help="Train NN after every 100 steps")
    parser.add_argument("--replay-fill", type=int, default=100000, help="number of elements in replay buffer before training starts")
    parser.add_argument("--reg-term", type=float, default=1e-3, help="multiplier before policy regularization term")
    parser.add_argument("--clip-term", type=float, default=0.5, help="gradient clipping parameter")
    parser.add_argument("--tau", type=float, default=0.02, help="target network update parameter")
    # Checkpointing
    
    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--report-every", type=int, default=500, help="report training stats every 500 episodes")
    return parser.parse_args()

##### Make Env #####
def make_env(display): 
    #make able to chose type of environment from directory based on arglist.scenario
    #from Environments.information_exchange.environment import multimodal
    from Environments.multi_target_consensus.environment import multimodal
    #make able to pass max-episode-len argument to environment
    env = multimodal(display) #create arguments to specify communication channels in env & type of action space
    return env

##### Replay Memory #####
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity, env, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self.env = env
    
    def push(self, state, action, reward, next_state, done):            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = torch.tensor(np.asarray(state).reshape((-1, 2*self.env.input_state_size)), dtype=torch.float, device=self.device)
        action = torch.tensor(np.asarray(action).reshape((-1, 2*self.env.action_size)), dtype=torch.float, device=self.device)
        reward = torch.tensor(np.asarray(reward), dtype=torch.float, device=self.device)
        next_state = torch.tensor(np.asarray(next_state).reshape((-1, 2*self.env.input_state_size)), dtype=torch.float, device=self.device)
        done = torch.tensor(np.asarray(done), dtype=torch.float, device=self.device)

        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
        
##### Actor #####
class Actor(nn.Module):
    def __init__(self,state_size,action_size,units):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size,units)
        self.fc2 = nn.Linear(units,units)
        self.fc2h = nn.Linear(units,units) #extra optional layer
        self.fc3 = nn.Linear(units,action_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2h(x)) #may comment this
        x = self.fc3(x)
        return x

##### Critic #####
class Critic(nn.Module):
    def __init__(self,state_size,action_size,units):
        super(Critic, self).__init__()        
        self.fc1 = nn.Linear(2*(state_size+action_size),units)
        self.fc2 = nn.Linear(units, units)
        self.fc2h = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, 1)
    def forward(self, x_a):
        out = F.relu(self.fc1(x_a))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc2h(out)) #may comment this
        out = self.fc3(out)

        return out.view(-1)

##### Agent #####
class Agent():
    def __init__(self, env, train_args, device):
        self.device = device
        state_size = env.input_state_size
        action_size = env.action_size
        self.num_acts = action_size-env.num_comm

        self.actor = Actor(state_size,action_size,train_args.num_units).to(self.device)
        self.actor_target = Actor(state_size,action_size,train_args.num_units).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_size,action_size,train_args.num_units).to(self.device)
        self.critic_target = Critic(state_size,action_size,train_args.num_units).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=train_args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=train_args.lr)

    def onehot_from_logits(self, logits):
        """
        Given batch of logits, return one-hot sample 
        """
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()

        return argmax_acs

    def gumbel_softmax(self, logits):
        """
        Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
        Returns:
          If discretize=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        eps=1e-20
        shape = logits.size()
        U = torch.rand(shape).to(self.device)
        y = F.softmax(logits -torch.log(-torch.log(U + eps) + eps), dim=1)

        y_hard = self.onehot_from_logits(y)
        y = (y_hard - y).detach() + y

        return y

    def sample(self, O, grad_mode, mode="normal", gumbel=False):
        with torch.set_grad_enabled(grad_mode):
            if mode == "normal": 
                logits = self.actor(O)
            else :
                logits = self.actor_target(O)
            if gumbel == True:
                action = torch.cat((self.gumbel_softmax(logits[:,:self.num_acts]), self.gumbel_softmax(logits[:,self.num_acts:])), 1)
            else:
                action = torch.cat((self.onehot_from_logits(logits[:,:self.num_acts]), self.onehot_from_logits(logits[:,self.num_acts:])), 1)

        return logits, action

    def soft_update(self, target, source, tau):
        """
        Perform soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

##### MADDPG #####
class MADDPG():
    def __init__(self, exp_name, train_args, env, batch_size, replay_size, tau, gamma):
        #self.device = torch.device("cuda:0")
        self.device = torch.device("cpu")

        self.exp_dir = os.path.join("log", exp_name)
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        self.model_dir = os.path.join(self.exp_dir, "models")

        self.inp_size = env.input_state_size
        self.out_size = env.action_size
        self.num_comm = env.num_comm
        self.train_args = train_args
        try :
            os.mkdir(self.exp_dir)
        except:
            pass
        try :
            os.mkdir(os.path.join(self.tensorboard_dir))
        except:
            pass
        configure(self.tensorboard_dir, flush_secs=0.5)

        try :
            os.mkdir(self.model_dir)
        except:
            pass

        self.batch_size = batch_size

        self.replay_size = int(replay_size)
        self.replay_buffer = ReplayBuffer(self.replay_size, env, self.device)
        self.tau = tau
        self.gamma = gamma

        self.agents = [Agent(env,train_args,self.device) for i in range(2)]
        self.critic_loss_fn = nn.MSELoss()

    def save_model(self, mode):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(self.model_dir, "actor_"+str(i)+ mode))        

    def load_models(self, mode):
        cpu_device = torch.device("cpu")
        for i, agent in enumerate(self.agents):
            agent.actor.to(cpu_device)
            agent.actor.load_state_dict(torch.load(os.path.join(self.model_dir, "actor_"+str(i)+ mode)))
            agent.actor.to(self.device).eval()
            
    def act(self, o, gumbel):
        a = []
        for i, agent in enumerate(self.agents):
            O = torch.tensor(np.array([o[i]]), dtype=torch.float, device=self.device)
            logits, action = agent.sample(O=O, grad_mode=False,mode="normal", gumbel=gumbel)
            a.append(action.cpu().numpy()[0])

        return a

    def train(self):

        for i, agent in enumerate(self.agents):
            O, A, R, O_1, D = self.replay_buffer.sample(self.batch_size)
            logits1, a_1_1 = self.agents[0].sample(O_1[:,:self.inp_size], grad_mode=False, mode="target", gumbel=True)
            logits2, a_1_2 = self.agents[1].sample(O_1[:,self.inp_size:], grad_mode=False, mode="target", gumbel=True)
            A_1 = torch.cat((a_1_1, a_1_2),1)
            # Critic Loss
            with torch.set_grad_enabled(False):
                next_q_value = agent.critic_target(torch.cat((O_1, A_1), 1))
                expected_q_value = R[:,i] + self.gamma * next_q_value * (1 - D)

            q_value = agent.critic(torch.cat((O, A), 1))
            critic_loss = self.critic_loss_fn(q_value, expected_q_value)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm(agent.critic.parameters(), self.train_args.clip_term)
            agent.critic_optimizer.step()

            logits, new_action = agent.sample(O=O[:,self.inp_size*i:self.inp_size*(i+1)], grad_mode=True, mode="normal", gumbel=True)

            if i == 0:
                A_mod = torch.cat((new_action, A[:, self.out_size:2*self.out_size]), 1) #because 2 agents
            else :
                A_mod = torch.cat((A[:,:self.out_size], new_action), 1)

            actor_loss = -torch.mean(agent.critic(torch.cat((O, A_mod), 1))) + (logits**2).mean() * self.train_args.reg_term

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm(agent.actor.parameters(), self.train_args.clip_term)
            agent.actor_optimizer.step()

        for agent in self.agents:
            #Update Targets
            agent.soft_update(agent.critic_target, agent.critic, self.tau)
            agent.soft_update(agent.actor_target, agent.actor, self.tau)

##### MAIN #####
def train(arglist):
    exp_name = "exp_3"
    best_total_reward = -np.inf
    avg = -1000
    from datetime import datetime as dt

  # Train
    update_count = 0
    max_episodes = 1000000
    st = dt.now()

    env = make_env(arglist.display) #create arguments to specify communication channels in env & type of action space
    maddpg = MADDPG(exp_name, arglist, env, batch_size=arglist.batch_size, replay_size=1e6, tau=arglist.tau, gamma=arglist.gamma)
    total_rewards = []
    for episode in range(arglist.num_episodes):
        total_reward = 0.0
        o,_,_,_ = env.reset()
        t = 0
        avg = np.mean(total_rewards[-arglist.report_every:]) #Average reward from last 500 episodes
        while True:
            a = maddpg.act(o, gumbel=True)
            o_1, r, done, info = env.step(a)
            t += 1
            maddpg.replay_buffer.push(o, a, r, o_1, int(done)) #Stash Experience to Replay Buffer
            total_reward += (r[0]+r[1])
            o = o_1
            if maddpg.replay_buffer.__len__() < arglist.replay_fill :
                pass
            else :
                if update_count == 0: #Update only every 100 steps
                    #print("Training....")
                    maddpg.train()
                    #total_rewards=[] #reset average reward list every 100 episodes
                update_count = (update_count+1)%arglist.update_every 
            if done:
                total_rewards.append(total_reward)
                break
        log_value('total_reward', total_reward, episode)
        log_value('avg', avg, episode)
        if episode % arglist.report_every == 0:
            maddpg.save_model("ckpt_"+str(episode))
            #total_rewards=[]
            tdiff = dt.now()-st
            print("Episode : ",episode, "Time taken in seconds:",tdiff.seconds)
            print(" Total Steps :", t, "info :",info, " Average Total Reward :", avg)
            st = dt.now()
        if avg > best_total_reward:
            #print("avg ",avg)
            #print("best_total_reward ",best_total_reward)
            best_total_reward = copy.copy(avg)
            maddpg.save_model("best")
            #print("Saved best models !")

def evaluate():
    exp_name = "exp_2"
    flag = False
    max_episodes = 1000
    M1 = 0
    M2 = 0
    avg = 0.0

    env = make_env(scenario_name="simple", display=False, use_seed=False)

    maddpg = MADDPG(exp_name, batch_size=1024, replay_size=1e6, tau=0.02, gamma=0.95)
    maddpg.load_models("best")
    print("Loaded best models !!!")

    for episode in range(max_episodes):
        total_reward = 0.0
        o = env.reset()
        # print(o)
        # print("----"*40)
        # # env.R.color_shape_info()
        # print("")
        t = 0
        while True:
            a = maddpg.act(o, gumbel=False)
            o_1, r, done, info = env.step(a)
            t += 1
            total_reward += r[0]
            o = o_1

            if done:
                print(" ")
                print("Episode : ",episode)
                print(" Total Steps :", t, "info :",info,  " Total Reward :", total_reward)
                if info[0]:
                    M1 += 1
                if info[1]:
                    M2 += 1
                avg += total_reward
                break
    M1 = (M1/max_episodes) * 100
    M2 = (M2/max_episodes) * 100
    print("-"*80)
    print("M1 :", M1,"M2 :", M2)
    print("Average total reward",avg/max_episodes)

if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)
    #evaluate()

# example run: python main.py --display (to display training in pygame window)
# python main.py (headless training)
'''
To do:
1. create arguments to specify communication channels in env & type of action space in env
2. generalize code for n agents- currently hardcoded for 2 agents
3. add multi target consensus (6 landmarks case) in the environment directory
4. make able to chose type of environment from directory based on arglist.scenario
5. make able to pass max-episode-len argument to environment
6. possibly spawn n different simulations in n parallel threads each writing their experience to a common replay buffer
   thus new experiences are encountered n times as fast and hence updates could be n times as fast (data parallel)
7. Use environments evaluate (provided with the step function) instead of this evaluate
'''