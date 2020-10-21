#https://nerdparadise.com/programming/pygame/part4
import pygame
import math
import time
import numpy as np
import sys
import time as t
import socket
import shapely
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import copy
import re

import random

np.random.seed(9)
# To do
# Reset should randomly change agent agent starting positions
# Reset should shuffle the positions of landmarks
# Add 2 more landmarks
# Add sparse random pixel noise on the map (?)
# OU process noise?

def _reset(lmlist,aglist,forb,info):
    locs,props=[],[]
    if info=="reset":
        for i in aglist:
            i.visited = []
        #place_randomly(lmlist,aglist,forb,[8,6])
        locs,props=place_randomly_everything(lmlist,aglist,gridding=[8,6])
        #print("props in _reset ",props)
    return locs,props
def place_randomly_everything(lmlist,aglist,gridding=[8,6]):
    #Each entity (agent or landmark) has 350/8 and 250/8 units to sit on
    numindices=int(gridding[0]*gridding[1])
    place_locs = np.random.choice(numindices, len(lmlist)+len(aglist), replace=False)
    colors1 = (255,0,0)
    colors2 = (0,255,0)
    colors3 = (0,0,255)
    #making sure atleast 2 different shapes would get the same color
    #a=[random.sample((colors1,colors2,colors3),2) for i in range(3)]
    #color_assign = a[0]+a[1]+a[2]

    a=random.sample((1,2,3),2)
    b=random.sample((1,2,3),3)
    c=list(np.setdiff1d(b,a))
    d=np.where(b==c[0])
    e=[int(d[0][0])+1]
    f=np.setdiff1d(b,e)-1
    if(e[0]==2):
        f=f[0]
    if(e[0]==1):
        f=f[1]
    try:
        g=random.sample(tuple(f),1)[0]
    except:
        g=f
    b.insert(g,c[0])
    color_assign = a+b


    properties = random.sample(([colors1,"circle"],[colors1,"square"],[colors1,"triang"],[colors2,"circle"],[colors2,"square"],[colors2,"triang"],[colors3,"circle"],[colors3,"square"],[colors3,"triang"]),9)
    properties = properties[:6]
    #print("In place_randomly_everything got target_prop ",target_prop)
    for i in range(len(lmlist)): #Place randomly for the landmarks
        p_x = (place_locs[i]%8)*43.7 +40.0
        p_y = (int(place_locs[i]/8))*41.6 +40.0
        #print("Landmark ",p_x,p_y)
        lmlist[i].base_pos=(int(p_x),int(p_y)) #Maybe later can add some random perturbations
        lmlist[i].orientation = np.random.uniform(0,(2*math.pi),1)
        #lmlist[i].color = color_assign[i]
        if(color_assign[i]==1):
            lmlist[i].color = colors1
        if(color_assign[i]==2):
            lmlist[i].color = colors2
        if(color_assign[i]==3):
            lmlist[i].color = colors3
        #lmlist[i].shape = properties[i][1]

        lmlist[i].size = np.random.choice([20,25,30])
    #Randomly assign a landmark as a target
    target_prop=[]
    for i in lmlist:
        i.target=False
    idx=np.random.choice(len(lmlist))
    for i in range(len(lmlist)):
        if(i==idx):
            lmlist[i].target=True
            #shape properties of target
            if(lmlist[i].shape=="circle"):
                target_prop.append([1.0,0.0,0.0])
            if(lmlist[i].shape=="square"):
                target_prop.append([0.0,1.0,0.0])
            if(lmlist[i].shape=="triang"):
                target_prop.append([0.0,0.0,1.0])
            #color properties of target
            target_prop.append([a/255 for a in lmlist[i].color])   
            break
    for i in range(len(aglist)): #6 landmarks and 2 agents
        p_x = (place_locs[i+len(lmlist)]%8)*43.7 + 40.0
        p_y = (int(place_locs[i+len(lmlist)]/8))*41.6 + 40.0
        #print("Agent ",p_x,p_y)
        aglist[i].base_pos=[int(p_x),int(p_y)] #Maybe later can add some random perturbations
        #aglist[i].orientation = np.random.uniform(0,(2*math.pi),1)
        aglist[i].orientation = math.pi/4
    return place_locs, target_prop

def place_randomly(lmlist,aglist,forbidden,gridding=[8,6]):
    #Each entity (agent or landmark) has 350/8 and 250/8 units to sit on
    numindices=int(gridding[0]*gridding[1])
    place_locs = forbidden
    aglocchoices = np.setdiff1d(np.arange(numindices), forbidden[:-2], assume_unique=False)
    place_locs[-2:] = np.random.choice(aglocchoices, len(aglist), replace=False) #Force the agent to respawn somewhere else
    colors1 = [50,100,150,200,250]
    colors2 = [50,100,150,200,250]
    colors3 = [50,100,150,200,250]
    #Do nothing with landmarks in this function, just randomly place the agents
    for i in range(len(aglist)): #6 landmarks and 2 agents
        p_x = (place_locs[i+len(lmlist)]%8)*43.7 + 40.0
        p_y = (int(place_locs[i+len(lmlist)]/8))*41.6 + 40.0
        print("Agent ",p_x,p_y)
        aglist[i].base_pos=[int(p_x),int(p_y)] #Maybe later can add some random perturbations
        #aglist[i].orientation = np.random.uniform(0,(2*math.pi),1)
        aglist[i].orientation = math.pi/4

def agentColAgent(agent1,agent2):
    return 0
    '''
    p1 = Polygon(agent1.points)
    p2 = Polygon(agent2.points)
    if p1.intersects(p2): #True or False
        return 1
    else:
        return 0
    '''
def agentColLandmark(agent,landmark):
    p1 = Polygon(agent.points)
    if(agent.base_pos[0]>370 or agent.base_pos[0]<30 or agent.base_pos[1]>270 or agent.base_pos[1]<30):
        return 1 #Agent out of the map
    idx = 0
    for i in landmark.points:
        ''' #Collision with circle or triangle needs to be rewritten
        if(len(i)==3):#The landmark is a circle
            p = Point(i[0],i[1])
            circle = p.buffer(i[2])
            if(p1.intersects(circle)==True):# and landmark.targets[idx] == False): #Dont penalize for colliding with target
                return 1
        if(len(i)==4):#The landmark is a square
            p2=Polygon(i)
            if(p1.intersects(p2)==True):
                return 1
        '''
        idx +=1
    return 0
def interpret_command(actions,data):
    # argmax mode of action
    act1 = np.argmax(data[0:5])
    act2 = np.argmax(data[5:10])
    move_val = 5.0

    if(act1==0):
        actions[0] = move_val
    if(act1==1):
        actions[1] = move_val
    if(act1==2):
        actions[0] = -move_val
    if(act1==3):
        actions[1] = -move_val
    if(act1==4):
        actions[1] = 0.0
        actions[0] = 0.0

    if(act2==0):
        actions[3] = move_val
    if(act2==1):
        actions[4] = move_val
    if(act2==2):
        actions[3] = -move_val
    if(act2==3):
        actions[4] = -move_val
    if(act2==4):
        actions[3] = 0.0
        actions[4] = 0.0
    '''
    # Continuous velocity mode of action
    actions[0] +=0.5*data[0] #Velocity increment x agent 1
    actions[1] +=0.5*data[1] #velocity increment y agent 1
    actions[0] -=0.5*data[2] #Velocity decrement x agent 1
    actions[1] -=0.5*data[3] #velocity decrement y agent 1
    if(data[4]>0.5): #provides a chance to absolute stop
        actions[0]=0.0
        actions[1]=0.0

    actions[3] +=0.5*data[5] #Velocity increment x agent 2
    actions[4] +=0.5*data[6] #velocity increment y agent 2
    actions[3] -=0.5*data[7] #Velocity decrement x agent 2
    actions[4] -=0.5*data[8] #velocity decrement y agent 2
    if(data[9]>0.5):
        actions[3]=0.0
        actions[4]=0.0
    #data[9] is the do nothing action
    '''
    #No rotation commands given this time
    return actions
def square_points(base_pos,diag,ang): #position, size and orientation
    #color = (128, 0, 128) # purple 
    points = []
    inc = (math.pi)/2
    points.append((base_pos[0]+math.cos(ang)*diag, base_pos[1]+math.sin(ang)*diag))
    points.append((base_pos[0]+math.cos(ang+inc)*diag, base_pos[1]+math.sin(ang+inc)*diag))
    points.append((base_pos[0]+math.cos(ang+2*inc)*diag, base_pos[1]+math.sin(ang+2*inc)*diag))
    points.append((base_pos[0]+math.cos(ang+3*inc)*diag, base_pos[1]+math.sin(ang+3*inc)*diag))
    return points
def tri_points(base_pos,diag,ang): #position, size and orientation
    #color = (128, 0, 128) # purple 
    points = []
    inc = (math.pi)/2
    points.append((base_pos[0]+math.cos(ang)*diag, base_pos[1]+math.sin(ang)*diag))
    points.append((base_pos[0]+math.cos(ang+inc)*diag, base_pos[1]+math.sin(ang+inc)*diag))
    points.append((base_pos[0]+math.cos(ang+2.5*inc)*diag, base_pos[1]+math.sin(ang+2.5*inc)*diag))
    #points.append((base_pos[0]+math.cos(ang+3*inc)*diag, base_pos[1]+math.sin(ang+3*inc)*diag))
    return points
def create_background(width, height):
    #colors = [(255, 255, 255), (212, 212, 212)] #Think before changing colors..collision is hardcoded on color
    colors = [(255, 255, 255), (255, 255, 255)] #Complete white background for now
    bcolor = (50, 50, 50)
    background = pygame.Surface((width, height))
    tile_width = 20
    y = 0
    while y < height:
        x = 0
        while x < width:
            row = y // tile_width
            col = x // tile_width
            pygame.draw.rect(
                    background, 
                    colors[(row + col) % 2],
                    pygame.Rect(x, y, tile_width, tile_width))
            x += tile_width
        y += tile_width
    #Add a thin border around the walls
    pygame.draw.rect(background, bcolor,pygame.Rect(0, 0, width/20, height))
    pygame.draw.rect(background, bcolor,pygame.Rect(0, 0, width, height/20))
    pygame.draw.rect(background, bcolor,pygame.Rect(19*width/20, 0, width/20, height))
    pygame.draw.rect(background, bcolor,pygame.Rect(0, 19*height/20, width, height/20))
    return background

def is_trying_to_quit(event):
    pressed_keys = pygame.key.get_pressed()
    alt_pressed = pressed_keys[pygame.K_LALT] or pressed_keys[pygame.K_RALT]
    x_button = event.type == pygame.QUIT
    altF4 = alt_pressed and event.type == pygame.KEYDOWN and event.key == pygame.K_F4
    escape = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
    return x_button or altF4 or escape

class Agent(object):
    def __init__(self,base_pos = [150,50],size = 15,orientation = (math.pi)/4,color = (128, 0, 128),cam_parameters = [(math.pi)/6,60]):
        #Properties of agents (They are small square shapes for now)
        #Position, Size, Orientation, Color, Actions, Camera parameters:(fov_angle, fov_distance), 1 step belief
        self.base_pos = base_pos
        self.start_pos = copy.copy(self.base_pos)
        self.size = size
        self.orientation = orientation
        self.start_ort = copy.copy(self.orientation)
        self.color = color # purple 
        t=(math.pi)/100 #necessary so that it does not detect its own boundary
        self.ar_size = int(math.ceil(2*cam_parameters[0]/t))
        #self.actions = []
        self.cam_parameters = cam_parameters
        self.belief = 0 #Neutral about whether current position is good or bad
        self.points = square_points(self.base_pos,self.size,self.orientation)
        self.backtrack_actions=[]
        self.visited = [] #placeholder to keep account of landmarks it had visited
    def resetpos(self):
        self.base_pos = copy.copy(self.start_pos)
        self.orientation = copy.copy(self.start_ort)
    def act(self,actions):
        #self.backtrack_actions.append(actions)
        self.base_pos[0]+=actions[0]
        self.base_pos[1]+=actions[1]
        self.orientation+=actions[2]
        #self.draw(surface)
        c,d=0.0,0.0
        #c,d=self.lidar_scan(surface) #Dont waste time scanning 
        return c,d
    def backtrack(self):
        try:
            actions=self.backtrack_actions.pop()
            self.base_pos[0]-=actions[0]
            self.base_pos[1]-=actions[1]
            self.orientation-=actions[2]
        except:
            print("Cannot backtrack anymore")
    def draw(self,surface,comm):
        diag=self.size #size of the agent
        base_pos = self.base_pos
        ang = self.orientation

        name1="Depth"
        name2="DF"
        if(self.color==(200,0,100)):
            name1 = "Color"
            name2 = "CF"
        #change the color of the agent based on the communication
        #transparency = comm[3]*255.0*10
        transparency = 255.0
        color = (comm[0]*transparency, comm[1]*transparency, comm[2]*transparency)
        #color = self.color # purple 
        self.points = square_points(base_pos,diag,ang)
        pygame.draw.polygon(surface, color, self.points)
        #Draw the belief point
        #pygame.draw.circle(surface, color, (int(belief_point[0]),int(belief_point[1])), 5) #Draw a small circe of radius 5

        #Label the agents  and their belief points by drawing text on them
        font = pygame.font.Font(None,20)
        text = font.render(name1, 1, (50, 60, 70))
        textpos = (int(base_pos[0]),int(base_pos[1]))
        surface.blit(text, textpos)

        #text = font.render(name2, 1, (50, 60, 70))
        #textpos = (int(belief_point[0]),int(belief_point[1]))
        #surface.blit(text, textpos)


        cam_capture=[]
        #fov_dist=60
        #fov_ang=(math.pi)/6
        fov_dist=self.cam_parameters[1]
        fov_ang=self.cam_parameters[0]
        if(fov_ang==math.pi):
            pygame.draw.circle(surface, color, (int(base_pos[0]),int(base_pos[1])), fov_dist+1,2)
        if(fov_ang!=math.pi):
            cam_capture.append((base_pos[0], base_pos[1]))
            cam_capture.append((base_pos[0]+math.cos(ang-fov_ang)*fov_dist, base_pos[1]+math.sin(ang-fov_ang)*fov_dist))
            cam_capture.append((base_pos[0]+math.cos(ang+fov_ang)*fov_dist, base_pos[1]+math.sin(ang+fov_ang)*fov_dist))
            pygame.draw.polygon(surface, color, cam_capture, 1)
    def lidar_scan(self,surface):
        '''
        Inputs:
        x,y position of the agent
        angle orientation of the agent
        camera field of view angle for the agents camera
        camera field of view distance for the agents camera
        size of the agent
        current screen where world is being rendered
        '''
        #Do a lidar scan
        R=self.cam_parameters[1]
        fov_ang=self.cam_parameters[0]
        rad=self.size
        base_pos=self.base_pos
        ang=self.orientation

        t=(math.pi)/100 #necessary so that it does not detect its own boundary
        ar_size = int(math.ceil(2*fov_ang/t))
        color_array=0.02*np.ones((self.ar_size,3))
        depth_array=0.02*np.ones((self.ar_size))
        
        while (t<2*fov_ang-(math.pi)/100):
            t+=(math.pi)/100
            while (rad<R):
                rad+=0.05
                q_point=(base_pos[0]+math.cos(ang-fov_ang+t)*rad, base_pos[1]+math.sin(ang-fov_ang+t)*rad)
                try:
                    r,g,b,a=surface.get_at((int(q_point[0]),int(q_point[1]))) #Do a query for the pixel value
                except:#scan got out of screen
                    depth_array[int(100*t/(math.pi))] = 0.02
                    color_array[int(100*t/(math.pi)),:]=[0.02,0.02,0.02]
                    continue
                if((r!=255 or g!=255 or b!=255) and (r!=128 and g!=0 and b!=128)): #Something is there, its not the camera fov contour
                   depth_array[int(100*t/(math.pi))]=math.sqrt((q_point[0]-base_pos[0])**2+(q_point[1]-base_pos[1])**2)
                   color_array[int(100*t/(math.pi)),:]=[r,g,b]
                   break #No need to find deeper points
            rad=self.size
        #print(color_array)
        #print(depth_array)
        depth_array=np.array(depth_array,dtype=np.int)
        color_array=np.array(color_array,dtype=np.int)
        #print("Depth array in lidar scan function ",depth_array)
        return color_array,depth_array

class LandMark(object):
    def __init__(self,shape="circle",size=30,base_pos=(200,150),orientation=(math.pi)/4,color=(0, 140, 255),target=True):
        #Properties of landmarks
        #Position, Size, Orientation, Color, Shape type (Circle or square)
        self.shape = shape
        self.size = size
        self.base_pos = (int(base_pos[0]),int(base_pos[1]))
        self.orientation = orientation
        self.color = color # aquamarine
        self.target = target
class LandMarks(object):
    def __init__(self,scrwidth,scrheight,landmarks=[]):
        self.landmarks=landmarks
        self.wrange=scrwidth
        self.hrange=scrheight
        self.points=[]
        self.targets = []
    def create(self):
        self.landmarks=[]
        shape = np.random.choice(["circle","square"],1)
        size = np.random.randint(10,100)
        base_pos = (np.random.randint(self.wrange,1),np.random.randint(self.hrange,1))
        orientation = np.random.uniform(0,(2*math.pi),1)
        color = (np.random.randint(255,1),np.random.randint(255,1),np.random.randint(255,1))
        lm=LandMark(shape,size,base_pos,orientation,color)
        self.landmarks.append(lm)
    def place(self):
        for l in self.landmarks:
            l.base_pos = (np.random.randint(self.wrange,1),np.random.randint(self.hrange,1))
            l.orientation = np.random.uniform(0,(2*math.pi),1)
    def target_landmark_position(self):
        for l in self.landmarks:
            if(l.target):
                return np.around([(l.base_pos[0]-200.0)/400.0,(l.base_pos[1]-150.0)/300.0,l.orientation],decimals=2)
    def all_landmark_position(self,agent1,agent2):
        a1_rp_c=[]
        a1_rp_s=[]
        a1_rp_t=[]

        a2_rp_r=[]
        a2_rp_g=[]
        a2_rp_b=[]

        for l in self.landmarks:
            code = []
            if(l.shape=="circle"):
                a1_rp_c.append(np.around([(agent1.base_pos[0]-l.base_pos[0])/400.0,(agent1.base_pos[1]-l.base_pos[1])/300.0], decimals=2))
            if(l.shape=="square"):
                a1_rp_s.append(np.around([(agent1.base_pos[0]-l.base_pos[0])/400.0,(agent1.base_pos[1]-l.base_pos[1])/300.0], decimals=2))
            if(l.shape=="triang"):
                a1_rp_t.append(np.around([(agent1.base_pos[0]-l.base_pos[0])/400.0,(agent1.base_pos[1]-l.base_pos[1])/300.0], decimals=2))
            
            if(l.color==(255,0,0)):
                a2_rp_r.append(np.around([(agent2.base_pos[0]-l.base_pos[0])/400.0,(agent2.base_pos[1]-l.base_pos[1])/300.0], decimals=2))
            if(l.color==(0,255,0)):
                a2_rp_g.append(np.around([(agent2.base_pos[0]-l.base_pos[0])/400.0,(agent2.base_pos[1]-l.base_pos[1])/300.0], decimals=2))
            if(l.color==(0,0,255)):
                a2_rp_b.append(np.around([(agent2.base_pos[0]-l.base_pos[0])/400.0,(agent2.base_pos[1]-l.base_pos[1])/300.0], decimals=2))
        
        agent1_rel_pos = a1_rp_c+ a1_rp_s+ a1_rp_t
        #print("agent1_rel_pos ",agent1_rel_pos)
        agent2_rel_pos = a2_rp_r+ a2_rp_g+ a2_rp_b
        #print("agent2_rel_pos ",agent2_rel_pos)

        return agent1_rel_pos + agent2_rel_pos #(should be of length 12+12)
    def draw(self,surface):
        self.points=[]
        for l in self.landmarks:
            if(l.shape=="circle"):
                pygame.draw.circle(surface, l.color, l.base_pos, l.size) #orientation doesnt matter for circle
                self.points.append([l.base_pos[0],l.base_pos[1],l.size])
            elif(l.shape=="square"):
                points = square_points(l.base_pos,l.size,l.orientation)
                self.points.append(points)
                pygame.draw.polygon(surface, l.color, points)
            elif(l.shape=="triang"):
                points = tri_points(l.base_pos,l.size,l.orientation)
                self.points.append(points)
                pygame.draw.polygon(surface, l.color, points)
            if(l.target == True):
                font = pygame.font.Font(None,20)
                text = font.render("Target", 1, (50, 60, 70))
                textpos = (int(l.base_pos[0]),int(l.base_pos[1]))
                surface.blit(text, textpos)


class Render(object):
    def __init__(self,width,height,display=True):
        #Initialize landscape and agents
        self.scrwidth=width
        self.scrheight=height
        self.display=display

        self.actions = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.numtries=0
        #passing FOV=pi to agent means 360 view 
        self.agents=[Agent([50,50],15,(math.pi)/4,(128, 0, 128),[(math.pi),30]),Agent([50,250],15,(math.pi)/4,(200, 0, 100),[(math.pi),30])]
        self.bps=[0.0,0.0,0.0,0.0]

        lm1=LandMark(shape="circle",size=20,base_pos=(200,150),orientation=(math.pi)/4,color=(255, 0, 0),target=False)
        lm2=LandMark(shape="circle",size=20,base_pos=(300,200),orientation=(math.pi)/4,color=(0, 255, 0),target=False)
        lm3=LandMark(shape="square",size=20,base_pos=(300,200),orientation=(math.pi)/4,color=(0, 0, 255),target=False)
        lm4=LandMark(shape="square",size=20,base_pos=(200,150),orientation=(math.pi)/4,color=(255, 0, 0),target=False)
        lm5=LandMark(shape="triang",size=20,base_pos=(300,200),orientation=(math.pi)/4,color=(0, 255, 0),target=False)
        lm6=LandMark(shape="triang",size=20,base_pos=(300,200),orientation=(math.pi)/4,color=(0, 0, 255),target=False)

        self.lms=LandMarks(width,height,[lm1,lm2,lm3,lm4,lm5,lm6])

        self.forb,self.target_prop = place_randomly_everything(self.lms.landmarks,self.agents,gridding=[8,6])
        pygame.init()
        if self.display:
            #Initialize rendering parameters
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Multimodal environment')
            self.background = create_background(width, height)
        

    def set_actions(self,agentidx,actions):
        c,d=self.agents[agentidx].act(actions,self.screen)
    def render_world(self,data,comm,reset_info):
        #action=[[150,50],(math.pi)/4]
        width=self.scrwidth
        height=self.scrheight
        c=[]
        d=[]
        actions = self.actions
        data = np.array(data,dtype=np.float32)
        actions=interpret_command(actions,data)


        c,d=self.agents[0].act(actions[0:3])
        e,f=self.agents[1].act(actions[3:6])
        
        if self.display:
            #For display only (does not affect training)
            self.screen.blit(self.background, (0, 0))
            self.lms.draw(self.screen) #Need to draw the landmarks first
            self.agents[0].draw(self.screen,comm[0:3])
            self.agents[1].draw(self.screen,comm[3:6])
            t.sleep(0.2)
            pygame.display.flip()
        
        collision = [0,0]
        agent1pos = np.around([(self.agents[0].base_pos[0]-200.0)/400.0,(self.agents[0].base_pos[1]-150.0)/300.0,self.agents[0].orientation],decimals=2)
        agent2pos = np.around([(self.agents[1].base_pos[0]-200.0)/400.0,(self.agents[1].base_pos[1]-150.0)/300.0,self.agents[1].orientation],decimals=2)
        positions = [agent1pos,agent2pos,self.lms.target_landmark_position()]
        positions_rel = self.lms.all_landmark_position(self.agents[0],self.agents[1])

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                None 
            if is_trying_to_quit(event):
                sys.exit(0)
        
        if reset_info=="reset":
            self.actions = [0.0,0.0,0.0,0.0,0.0,0.0]
        forb,target_prop=_reset(self.lms.landmarks,[self.agents[0],self.agents[1]],self.forb,reset_info)
        if(len(target_prop)>0):
            self.target_prop = target_prop
        if(len(forb)>0):
            self.forb = forb

        return collision,positions,positions_rel,self.target_prop


