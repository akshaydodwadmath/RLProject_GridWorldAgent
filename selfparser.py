import json
import os
import numpy as np

pregrid = ['north', 'west', 'south', 'east']
wall = 4
postgrid = ['null','null','null','null','null','north', 'west', 'south', 'east']

action_list = ['move', 'turnLeft','turnRight','finish']

class GridWorld():
    def generate_bmpfeatures(self,inputlist):
        bitmaps = np.zeros((9, inputlist[0],  inputlist[1]))
        
        #Pre-Grid
        if( inputlist[4] == 'north'):
            bitmaps[pregrid.index('north'), inputlist[2],  inputlist[3]] = 1  
        elif( inputlist[4] == 'west'):
            bitmaps[pregrid.index('west'), inputlist[2],  inputlist[3]] = 1  
        elif( inputlist[4] == 'south'):
            bitmaps[pregrid.index('south'), inputlist[2],  inputlist[3]] = 1  
        elif( inputlist[4] == 'east'):
            bitmaps[pregrid.index('east'), inputlist[2],  inputlist[3]] = 1  
        
        #Walls
        for index in inputlist[8]:
            bitmaps[wall, index[0], index [1]] = 1
        
        #Post-Grid
        if( inputlist[7] == 'north'):
            bitmaps[postgrid.index('north'), inputlist[5],  inputlist[6]] = 1  
        elif( inputlist[7] == 'west'):
            bitmaps[postgrid.index('west'), inputlist[5],  inputlist[6]] = 1  
        elif( inputlist[7] == 'south'):
            bitmaps[postgrid.index('south'), inputlist[5],  inputlist[6]] = 1  
        elif( inputlist[7] == 'east'):
            bitmaps[postgrid.index('east'), inputlist[5],  inputlist[6]] = 1  
        #flatbitmaps = np.ndarray.flatten(bitmaps)
        #print(flatbitmaps)
        return bitmaps

    def next_state(self,state, action,steps):
        rew = -1
        #print(action_list[action])
        crash = False
        index_z, index_y,index_x = np.where(state[0:len(pregrid)] == 1)
        
        prev_index_z = index_z
        #print ("current_pos",index_z, index_y,index_x)
        if( action_list[action] != 'finish'):
            state[index_z, index_y,index_x] = 0
        
        if( action_list[action] == 'move'):
            if( index_z == pregrid.index('north')): #north
                index_y = index_y - 1
            elif( index_z == pregrid.index('south')): #south
                index_y = index_y + 1
            elif( index_z == pregrid.index('west')): #west
                index_x = index_x - 1
            elif( index_z == pregrid.index('east')): #east
                index_x = index_x + 1
            if(index_x > 3 or index_y > 3):
                #rew = -1
                crash = True
                state[0:len(pregrid), :,:] =np.zeros((len(pregrid), 4,  4))
                print("crash")
            elif(index_x < 0 or index_y < 0):
               # rew = -1
                crash = True
                state[0:len(pregrid), :,:] =np.zeros((len(pregrid), 4,  4))
                print("crash")
            elif(state[wall,index_y,index_x] == 1):
               # rew = -1
                crash = True
                state[0:len(pregrid), :,:] =np.zeros((len(pregrid), 4,  4))
                print("crash: Wall")
            else:
                state[index_z,index_y,index_x] = 1
        if( action_list[action] == 'turnLeft'):
            if( index_z == pregrid.index('north')): #north
                index_z = pregrid.index('west')
            elif( index_z == pregrid.index('south')): #south
                index_z = pregrid.index('east')
            elif( index_z == pregrid.index('west')): #west
                index_z = pregrid.index('south')
            elif( index_z == pregrid.index('east')): #east
                index_z = pregrid.index('north')
            state[index_z,index_y,index_x] = 1
        if( action_list[action] == 'turnRight'):

            if( index_z == pregrid.index('north')): #north
                index_z = pregrid.index('east')
            elif( index_z == pregrid.index('south')): #south
                index_z = pregrid.index('west')
            elif( index_z == pregrid.index('west')): #west
                index_z = pregrid.index('north')
            elif( index_z == pregrid.index('east')): #east
                index_z = pregrid.index('south')
            state[index_z,index_y,index_x] = 1
        
        #print("next_pos",index_z, index_y,index_x)    
        postindex_z, postindex_y,postindex_x = np.where(state[5:len(postgrid)] == 1)
       # print("post_pos",postindex_z, postindex_y,postindex_x)    

        if(action_list[action] == 'finish'):
            if(np.array_equiv(state[0:len(pregrid)], state[5:len(postgrid)])):
                print("reward of 10",action_list[action])
                state[0:len(pregrid), :,:] =np.ones((len(pregrid), 4,  4))
                rew = 10
                crash = True
            else:
                crash = True
                state[0:len(pregrid), :,:] =np.zeros((len(pregrid), 4,  4))
                print("crash: Finish")
        if(np.array_equiv(state[0:len(pregrid)], state[5:len(postgrid)])):
            rew = 5
            print("reward of 5",action_list[action])
            #crash = True
               # rew = -1
        #elif(crash == False):
         #   if((prev_index_z != index_z) and (postindex_z == index_z)):
          #      print("reward of 1")
           #     rew = 1
        
        #else:
           # print("no reward")
         #   rew = -0.1
            
        return state, rew, crash


    def generate_env(self,episode):
        root_fd = 'datasets/data_easy/train/train/task'
        print(episode)
        file_name = str(episode%4000) + '_task.json'
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
        root_fd = 'datasets/data_easy/train/train/seq'
        #print(episode)
        file_name = str(episode%4000) + '_seq.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            name = distro[0]
            actions = distros_dict[distro]
            temp_list.append(distros_dict[distro])
            ##print(distros_dict[distro])

        
        #state = self.generate_bmpfeatures(temp_list)
        return actions
        
    def generate_val(self,episode):
        root_fd = 'datasets/data_easy/val/task'
        print(episode)
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
        root_fd = 'datasets/data_easy/val/seq'
        #print(episode)
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
    # print(state)
    # next_state(state, 'move')
    # next_state(state, 'turnLeft')
    # next_state(state, 'move')
    # next_state(state, 'turnRight')
    # next_state(state, 'turnLeft')
    # next_state(state, 'turnLeft')
    # _,rew1 = next_state(state, 'move')
    # _,rew2 = next_state(state, 'turnRight')
    # print(rew1)
    # print(rew2)



