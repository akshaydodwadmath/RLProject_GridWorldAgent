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
        
        #bitmap_pre_pos = np.zeros((inputlist[0],  inputlist[1]))
        #bitmap_pre_ori = np.zeros((4, 1))
  
        #bitmap_post_pos = np.zeros((inputlist[0],  inputlist[1]))
        #bitmap_post_ori = np.zeros((4, 1))
        
        #bitmap_walls = np.zeros((inputlist[0],  inputlist[1]))
        
        #bitmap_pre_mark = np.zeros((inputlist[0],  inputlist[1]))
        #ssbitmap_post_mark = np.zeros((inputlist[0],  inputlist[1]))
        
        
        #Pre-Grid Location
        bitmaps4x4[pregrid_pos, inputlist[2], inputlist [3]] = 1
        #bitmaps.append(bitmap_pre_pos)
            
        #Pre-Grid Orientation
        if( inputlist[4] == 'north'):
            bitmaps4x1[pregrid_ori, 0] = 1  
        elif( inputlist[4] == 'west'):
            bitmaps4x1[pregrid_ori, 1] = 1  
        elif( inputlist[4] == 'south'):
            bitmaps4x1[pregrid_ori, 2] = 1  
        elif( inputlist[4] == 'east'):
            bitmaps4x1[pregrid_ori, 3]  = 1  
        #bitmaps.append(bitmap_pre_ori) 
        
        #Post-Grid Location
        bitmaps4x4[postgrid_pos, inputlist[5], inputlist [6]] = 1
        #bitmaps.append(bitmap_post_pos)
        
        #Post-Grid Orientation
        if( inputlist[7] == 'north'):
            bitmaps4x1[postgrid_ori, 0] = 1  
        elif( inputlist[7] == 'west'):
            bitmaps4x1[postgrid_ori, 1] = 1  
        elif( inputlist[7] == 'south'):
            bitmaps4x1[postgrid_ori, 2] = 1  
        elif( inputlist[7] == 'east'):
            bitmaps4x1[postgrid_ori, 3]  = 1  
        #bitmaps.append(bitmap_post_ori)
        
        #Walls
        for index in inputlist[8]:
            bitmaps4x4[wall, index[0], index [1]] = 1
       # bitmaps.append(bitmap_walls)
        
        for index in inputlist[9]:
            bitmaps4x4[ pregrid_mark, index[0], index [1]] = 1
       # bitmaps.append(bitmap_pre_mark)
            
        for index in inputlist[10]:
            bitmaps4x4[ postgrid_mark, index[0], index [1]] = 1
        #bitmaps.append(bitmap_post_mark)

        #flatbitmaps = np.ndarray.flatten(bitmaps)
        #print(flatbitmaps)
        return bitmaps4x4, bitmaps4x1

    def next_state(self,state1 , state2, action,steps):
        global flag
        rew = 0
        #print(action_list[action])
        crash = False
        finish = False
        index_y,index_x = np.where(state1[pregrid_pos, :, :] == 1)
        index_z, index_z2 = np.where(state2[pregrid_ori,:, :] == 1)
    #    print('indexz',index_z, index_z2)
        #prev_index_z = index_z
        #print ("current_pos",index_z, index_y,index_x)
     #   if( action_list[action] != 'finish'):
        
        
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
             #   rew = -0.1
                crash = True
                #state1[pregrid_pos, :,:] =np.zeros( (4,  4))
                #state1[pregrid_pos, 0,:] =np.ones( (4, ))
                print("crash")
                #rew = -50
            elif(index_x < 0 or index_y < 0):
            #    rew = -0.1
                crash = True
                #state1[pregrid_pos, :,:] =np.zeros((4,  4))
                #state1[pregrid_pos, 1,:] =np.ones( (4, ))
                print("crash")
               # rew = -50
            elif(state1[wall,index_y,index_x] == 1):
             #   rew = -0.3
                crash = True
                #state1[pregrid_pos,index_y,index_x] = 1
              #  state1[pregrid_pos, :,:] =np.zeros(( 4,  4))
                print("crash: Wall")
              #  rew = -50
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
                #if(np.array_equiv(state1[pregrid_mark, index_y, index_x], state1[postgrid_mark, index_y, index_x])):
                #    rew =  100
            else:
             #   rew = -0.3
                crash = True
              #  state1[pregrid_pos, :,:] =np.zeros(( 4,  4))
             #   state1[pregrid_pos, 2,:] =np.ones( (4, ))
                
                print("crash: pickMarker")
                
        if( action_list[action] == 'putMarker'):
            if(state1[pregrid_mark, index_y, index_x]  == 0): 
                state1[pregrid_mark, index_y, index_x] = 1
                #if(np.array_equiv(state1[pregrid_mark, index_y, index_x], state1[postgrid_mark, index_y, index_x])):
                #    rew =  100
            else:
              #  rew = -0.3
                crash = True
              #  state1[pregrid_pos, :,:] =np.zeros(( 4,  4))
             #   state1[pregrid_pos, 3,:] =np.ones( (4, ))
                print("crash: putMarker")
      #  print(state1)
      #  print(state2)
        
        #print("next_pos",index_z, index_y,index_x)    
       # postindex_z, postindex_y,postindex_x = np.where(state[5:len(postgrid)] == 1)
       # print("post_pos",postindex_z, postindex_y,postindex_x)    

        #if(flag == 1) and (action_list[action] != 'finish'):
         #   rew = -1
            
      #  if(action_list[action] == 'finish'):
            #if(np.array_equiv(state[0:len(pregrid)], state[5:len(postgrid)])):
      #      print("Finish",action_list[action])
      #      state[0:len(pregrid), :,:] =np.ones((len(pregrid), 4,  4))          
            #rew = 10
      #      crash = True
            #else:
            #    crash = True
                #rew = -5
            #    state[0:len(pregrid), :,:] =np.zeros((len(pregrid), 4,  4))
            ##    print("crash: Finish")
        if((np.array_equiv(state1[pregrid_pos], state1[postgrid_pos])) and (np.array_equiv(state2[pregrid_ori], state2[postgrid_ori]))):
            rew = 1
            print("reward of 100 ",action_list[action])
          #  state1[pregrid_pos, :,:] =np.ones(( 4,  4))
            #flag = 1
            finish = True
            crash = True
               # rew = -1
        #elif(crash == False):
         #   if((prev_index_z != index_z) and (postindex_z == index_z)):
          #      print("reward of 1")
           #     rew = 1
        
        #else:
           # print("no reward")
         #   rew = -0.1
            
        return state1, state2, rew, crash, finish


    def generate_env(self,episode):
        global flag
        flag = 0
        root_fd = 'datasets/data_medium/train/train/task'
        
        
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
    

    def generatelabel_env(self,episode):
        root_fd = 'datasets/data_medium/train/train/seq'
        #print(episode)
        file_name = str(episode) + '_seq.json'
        file_path = os.path.join(root_fd, file_name)
        temp_list = []
        with open(file_path, 'r') as f:
            distros_dict = json.load(f)
        myvars = {}
        for distro in distros_dict:
            
            name = distro[0]
            temp_list = distros_dict[distro]
            #print(temp_list)
            #temp_list.append(distros_dict[distro])
            ##print(distros_dict[distro])
        temp = temp_list.index('finish')
        actions = temp_list[:temp]
    
       # print(actions)
        
        #state = self.generate_bmpfeatures(temp_list)
        return actions
        
    def generate_val(self,episode):
        root_fd = 'datasets/data_medium/val/val/task'
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
        root_fd = 'datasets/data_medium/val/val/seq'
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



