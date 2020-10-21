from Environments.collaborative_localization.Rendering import Render
#from headless.Rendering_policy_viz import Render
import time as t
import re
import numpy as np
from collections import deque
import copy
import math
import sys

from multiagent.multi_discrete import MultiDiscrete
from gym import spaces

Poslist = []
def scale_x(X):
    return float((X-200.0)/400.0)
def scale_y(Y):
    return float((Y-150.0)/300.0)
def one_hot_encode(array):
    comm = np.zeros((len(array),))
    word = np.argmax(array)
    comm[word]=1.0
    #print("one hot encode ",comm)
    return comm
def decode_from_string(state):
    poslist = re.findall(r"[-+]?\d*\.\d+|\d+", state)
    Poslist = poslist
    if len(poslist)==200:
        a = np.array(poslist,dtype=np.float32)
        return a.reshape((200,1)) #Last number denoting channels
    if len(poslist)==200*3:
        a = np.array(poslist,dtype=np.float32)
        return a.reshape((200,3))
    if len(poslist)==9:
        b = np.array(poslist,dtype=np.float32)
        return b.reshape((3,3))
    if len(poslist)==6:
        b = np.array(poslist,dtype=np.float32)
        return b.reshape((2,3))
    if len(poslist)==30: #3*5 + 3*5
        a = np.array(poslist,dtype=np.float32)
        return a.reshape((2,15))
    if len(poslist)==2:
        return np.array(poslist,dtype=np.int)
    else:
        print("wrong poslist ",state)
    #print("decode_from_string returned nothing")
def decide_rewards(pos,prev_pos,cols,footprint1,footprint2,numsteps,num):
    goal_reached = False
    rew = -float(agdistances(pos)) #The goal this time is for agents to meet with each other
    # Exporation reward
    try:
        if(num==0):
            rew +=0.0*len(np.unique(footprint1,axis=0))/numsteps
        if(num==1):
            rew +=0.0*len(np.unique(footprint2,axis=0))/numsteps
    except:
        #print("could not reshape footprint ")
        pass
    #target in sight reward
    #if(distances(pos)[0]<=0.15 and distances(pos)[1]<=0.15): #close enough to the target
    #if(distances(pos)[0]<=0.15): #close enough to the target
    if(rew>=-0.15): #agents touch each other
        #rew +=1.0 #target object is in sight (give high reward)
        goal_reached = True
    #No collision reward
    if(cols[num]==1):
        #print("Collided !")
        rew = rew - 0.0 #Dont Discourage collisions
    return rew, goal_reached

def info():#Just a placeholder function
    return {'n':[{},{}]} #Imitating the info structure of multi particle environment  
def encode_to_string(action): #6 possible actions for an agent
    actstring=""
    for i in action:
        if i==0:
            actstring=actstring+"W"
        if i==1:
            actstring=actstring+"A"
        if i==2:
            actstring=actstring+"S"
        if i==3:
            actstring=actstring+"D"
        if i==4:
            actstring=actstring+"R"
        if i==5:
            actstring=actstring+"T"
    return actstring
def distances(pos):
    ag1pos=pos[0]
    ag2pos=pos[1]
    targetpos=pos[2]
    a1dist=math.sqrt((ag1pos[0]-targetpos[0])**2 + (ag1pos[1]-targetpos[1])**2) #85000 is the initial distance at reset
    a2dist=math.sqrt((ag2pos[0]-targetpos[0])**2 + (ag2pos[1]-targetpos[1])**2) #65000 is the initial distance at reset
    return [a1dist,a2dist]
def agdistances(pos):
    ag1pos=pos[0]
    ag2pos=pos[1]
    targetpos=pos[2]
    dist=math.sqrt((ag1pos[0]-ag2pos[0])**2 + (ag1pos[1]-ag2pos[1])**2) #85000 is the initial distance at reset
    return dist
def done(numcols,numtries):
    #if numtries>200 or numcols>40:
    if numtries>60:
        return True
    return False
class multimodal(object):
    def __init__(self,display=True):
        #Setup bindings for being able to use as a gym environment
        act_space1 = MultiDiscrete([[0,4],[0,2]]) #action space of agent 1 consisting of 5 movement actions and 3 communications
        act_space2 = MultiDiscrete([[0,4],[0,2]]) #action space of agent 1 consisting of 5 movement actions and 3 communications
        self.action_space = [act_space1,act_space2] #combined actions space of both the agents
        self.display = display
        #Observation of an agent:
        #   1. Its own x and y velocity (2)
        #   2. Relative distances from landmarks (3*5) and their nature (one hot encoded) (assume perfect detection) 
        #       relative distances are only activated once the agent has actually went close enough to the object
        #       if the agent had not gone close to the object a placeholder like [-1,-1,-1] is used
        #       relative distances are arranged in a common order for both the agents
        #   4. Target color code for depth agent (3) / Target shape code for color agent (3) (one hot encoded)
        #   5. 10 communication channels (10)
        self.observation_space = []
        self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(18,), dtype=np.float32))
        self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(18,), dtype=np.float32))

        self.input_state_size = 18
        self.action_size = 5+3
        self.num_comm = 3
        self.n = 2 #2 agents


        self.prev_pos = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
        self.prev_pos_rel = []
        self.prev_rew=[0.0,0.0]
        self.numcols = 0
        self.numtries = 0

        self.reset_string="go on"

        self.R=Render(400, 300, self.display)

    def initialize_belief(self):
        for i in range(self.Kstep):
            self.ag1belief.append([-0.5,-0.5,-0.5,0.0])
            self.ag1belief.append([-0.5,-0.5,-0.5,0.0])

    def reset(self):
        self.footprint1=[]
        self.footprint2=[]
        self.numcols=0
        #self.initialize_belief()
        self.R.render_world([0,0,0,0,1,0,0,0,0,1],[0,0,0,0,0,0],"reset")
        return self.random_explore(1)

    def step(self,actions,random_explore=False): #Assume the last bit of the action array contains the belief (goodness of location)
        action1 = actions[0]
        action2 = actions[1]

        action = np.concatenate((action1[:5],action2[:5])) #5 discreet movement actions each for agent 1 and agent 2 
        comm = np.concatenate((action1[5:],action2[5:]))
        #print("sending actions ",action)
        self.cols,self.pos,self.pos_rel,self.target_prop = self.R.render_world(action,comm,"go on")

        self.target_prop = np.array(self.target_prop).reshape((3,3))
        self.pos = np.array(self.pos).reshape((3,3))
        self.pos_rel = np.array(self.pos_rel).reshape((2,15))
        self.cols = np.array(self.cols)

        self.numtries+=1

        r1,goal_reached = decide_rewards(self.pos,self.prev_pos,self.cols,self.footprint1,self.footprint2,self.numtries,0)
        r2,goal_reached = decide_rewards(self.pos,self.prev_pos,self.cols,self.footprint1,self.footprint2,self.numtries,1)

        reward = [0.5*(r1+r2),0.5*(r2+r1)] #Shared reward
        #reward = [r1,r2] #Shared reward

        self.footprint1.append([self.pos[0][0],self.pos[0][1]])
        self.footprint2.append([self.pos[1][0],self.pos[1][1]])
        self.base_dist = [(self.pos[0][0]-50)**2 + (self.pos[0][1]-50)**2, (self.pos[1][0]-50)**2 + (self.pos[1][1]-250)**2]

        Done = False
        if done(self.numcols,self.numtries) or goal_reached:
            self.numtries = 0
            self.numcols = 0
            self.footprint1=[]
            self.footprint2=[]
            #print("Done !")
            #print("self.target_prop ",self.target_prop)
            Done = True
        else:
            self.reset_string = "go on"

        obs_ag1 = []
        obs_ag1.extend(self.pos_rel[0])
        obs_ag1.extend(action2[5:]) #last 3 communication channels of agent 2
        obs_ag1=np.array(obs_ag1)

        obs_ag2 = []
        obs_ag2.extend(self.pos_rel[1])
        obs_ag2.extend(action1[5:]) #last 3 communication channels of agent 1
        obs_ag2=np.array(obs_ag2)

        self.prev_pos_rel=self.pos_rel
        self.prev_rew=[r1,r2]
        self.prev_pos = copy.copy(self.pos)

        return [obs_ag1,obs_ag2],reward, Done, info()
        
    def random_explore(self,tries):
        s,r,done,info = [],[],[],[]
        for i in range (tries): #20 random exploration
            action1=np.array([0,0,0,0,1,0,0,0]) #both agents start by halt
            action2=np.array([0,0,0,0,1,0,0,0])
            action = np.array([action1,action2])
            s,r,done,info = self.step(action,True)
            #t.sleep(0.05)
        return s,r,done,info

    def end(self):
        self.c.close() #Close the server connection

if __name__ =='__main__':
    m=multimodal()
    m.random_explore(20)
