import json
import os
import numpy as np
import random 
pregrid_pos = 0
postgrid_pos = 1
wall = 2
pregrid_mark = 3
postgrid_mark = 4

pregrid_ori = 0
postgrid_ori = 1

directions = ['north', 'west', 'south', 'east']

postgrid = ['null','null','null','null','null','north', 'west', 'south', 'east']

action_list = ['move', 'turnLeft','turnRight', 'pickMarker', 'putMarker', 'finish']
flag = 0
class GridWorld():
    def generate_bmpfeatures(self,inputlist):
        bitmaps4x4 = np.zeros((5, inputlist[0],  inputlist[1]))
        bitmaps4x1 = np.zeros((2, 4, 1))
  
        #Pre-Grid Location
        bitmaps4x4[pregrid_pos, inputlist[2], inputlist [3]] = 1
            
        #Pre-Grid Orientation
        if( inputlist[4] == 'north'):
            bitmaps4x1[pregrid_ori, 0] = 1  
        elif( inputlist[4] == 'west'):
            bitmaps4x1[pregrid_ori, 1] = 1  
        elif( inputlist[4] == 'south'):
            bitmaps4x1[pregrid_ori, 2] = 1  
        elif( inputlist[4] == 'east'):
            bitmaps4x1[pregrid_ori, 3]  = 1  
        
        #Post-Grid Location
        bitmaps4x4[postgrid_pos, inputlist[5], inputlist [6]] = 1
        
        #Post-Grid Orientation
        if( inputlist[7] == 'north'):
            bitmaps4x1[postgrid_ori, 0] = 1  
        elif( inputlist[7] == 'west'):
            bitmaps4x1[postgrid_ori, 1] = 1  
        elif( inputlist[7] == 'south'):
            bitmaps4x1[postgrid_ori, 2] = 1  
        elif( inputlist[7] == 'east'):
            bitmaps4x1[postgrid_ori, 3]  = 1  
        
        #Walls
        for index in inputlist[8]:
            bitmaps4x4[wall, index[0], index [1]] = 1
        
        for index in inputlist[9]:
            bitmaps4x4[ pregrid_mark, index[0], index [1]] = 1
            
        for index in inputlist[10]:
            bitmaps4x4[ postgrid_mark, index[0], index [1]] = 1

        return bitmaps4x4, bitmaps4x1

    def next_state(self,state1 , state2, action,steps):
        global flag
        rew = 0
        crash = False
        finish = False
        index_y,index_x = np.where(state1[pregrid_pos, :, :] == 1)
        index_z, index_z2 = np.where(state2[pregrid_ori,:, :] == 1)

        if( action_list[action] == 'move'):
            state1[pregrid_pos, index_y,index_x] = 0
            if( index_z == directions.index('north')): #north
                index_y = index_y - 1
            elif( index_z == directions.index('south')): #south
                index_y = index_y + 1
            elif( index_z == directions.index('west')): #west
                index_x = index_x - 1
            elif( index_z == directions.index('east')): #east
                index_x = index_x + 1
            if(index_x > 3 or index_y > 3):
                crash = True
            elif(index_x < 0 or index_y < 0):
                crash = True
            elif(state1[wall,index_y,index_x] == 1):
                crash = True
            else:
                state1[pregrid_pos,index_y,index_x] = 1
        if( action_list[action] == 'turnLeft'):
            state2[pregrid_ori,index_z, 0] = 0
            if( index_z == directions.index('north')): #north
                index_z = directions.index('west')
            elif( index_z == directions.index('south')): #south
                index_z = directions.index('east')
            elif( index_z == directions.index('west')): #west
                index_z = directions.index('south')
            elif( index_z == directions.index('east')): #east
                index_z = directions.index('north')
            state2[pregrid_ori,index_z, 0] = 1
        if( action_list[action] == 'turnRight'):
            state2[pregrid_ori,index_z, 0] = 0
            if( index_z == directions.index('north')): #north
                index_z = directions.index('east')
            elif( index_z == directions.index('south')): #south
                index_z = directions.index('west')
            elif( index_z == directions.index('west')): #west
                index_z = directions.index('north')
            elif( index_z == directions.index('east')): #east
                index_z = directions.index('south')
            state2[pregrid_ori,index_z, 0] = 1
        
        if( action_list[action] == 'pickMarker'):
            if(state1[pregrid_mark, index_y, index_x]  == 1): 
                state1[pregrid_mark, index_y, index_x] = 0
            else:
                crash = True
            
                
        if( action_list[action] == 'putMarker'):
            if(state1[pregrid_mark, index_y, index_x]  == 0): 
                state1[pregrid_mark, index_y, index_x] = 1
            else:
                crash = True


        if((np.array_equiv(state1[pregrid_pos], state1[postgrid_pos])) and
        (np.array_equiv(state1[pregrid_mark], state1[postgrid_mark])) and
         (np.array_equiv(state2[pregrid_ori], state2[postgrid_ori]))):
            rew = 1
            finish = True
            crash = True
            
        return state1, state2, rew, crash, finish


    def generate_env(self,episode):
        global flag
        flag = 0
        root_fd = 'datasets/data/train/train/task'
       
        file_name = str(episode) + '_task.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            temp_list.append(distros_dict[distro])

        state = self.generate_bmpfeatures(temp_list)
        return state
    

    def generatelabel_env(self,episode):
        root_fd = 'datasets/data/train/train/seq'
        file_name = str(episode) + '_seq.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            temp_list = distros_dict[distro]
        temp = temp_list.index('finish')
        actions = temp_list[:temp]
    
        return actions
        
    def generate_val(self,episode):
        root_fd = 'datasets/data/val/val/task'
        file_name = str(episode) + '_task.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            temp_list.append(distros_dict[distro])

        state = self.generate_bmpfeatures(temp_list)
        return state

    def generatelabel_val(self,episode):
        root_fd = 'datasets/data/val/val/seq'
        file_name = str(episode) + '_seq.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            actions = distros_dict[distro]
            temp_list.append(distros_dict[distro])

        return actions



