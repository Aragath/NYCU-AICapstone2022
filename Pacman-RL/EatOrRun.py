#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# if supermode and ghost near, chase the ghost, else the same as normal
# if ghost is inside a certain distance, escape to its opposite, returns list of direction to go
# elif check if the ghost is chasing us, if not, return no control, or else escape 
# else return an empty list
# 0: left, 1: right, 2: up, 3: down 4: no control
def EatOrRun(playerState, ghostState) -> list:
    # get all pos
    g1 = ghostState[0]
    g2 = ghostState[1]
    g3 = ghostState[2]
    g4 = ghostState[3]
    player_pos = [playerState[0], playerState[1]]

    # get all dis between
    d1 = [coord1 - coord2 for (coord1, coord2) in zip(player_pos, g1)]
    d1 = abs(d1[0]) + abs(d1[1])
    d2 = [coord1 - coord2 for (coord1, coord2) in zip(player_pos, g2)]
    d2 = abs(d2[0]) + abs(d2[1])
    d3 = [coord1 - coord2 for (coord1, coord2) in zip(player_pos, g3)]
    d3 = abs(d3[0]) + abs(d3[1])
    d4 = [coord1 - coord2 for (coord1, coord2) in zip(player_pos, g4)]
    d4 = abs(d4[0]) + abs(d4[1])
    
    # get the min dis
    dis_list = [d1, d2, d3, d4]
    nearest_dis = min(dis_list)
    nearest_index = dis_list.index(nearest_dis) # if g1 is the nearest, index = 0
        
    now_pos = ghostState[nearest_index]
    x_dir = now_pos[0] - prev_pos[nearest_index][0]
    y_dir = now_pos[1] - prev_pos[nearest_index][1]
    x_between = playerState[0] - now_pos[0]
    y_between = playerState[1] - now_pos[1]
    x_samesign = (((x_dir < 0) == (x_between < 0)) or ((x_dir > 0) == (x_between > 0)))
    y_samesign = (((y_dir < 0) == (y_between < 0)) or ((y_dir > 0) == (y_between > 0)))
    
    next_step = []
    
    # if supermode and ghost near, chase the ghost
    if(playerState[3] > 0 and dis_to_chase * 25 > nearest_dis):
        if(x_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 0))):
            next_step.append(0)
        elif(x_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 1))):
            next_step.append(1)
        if(y_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 2))):
            next_step.append(2)
        elif(y_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 3))):
            next_step.append(3)            
        return next_step        
    
    # if no ghost is near, return empty list
    if(dis_between * 25 < nearest_dis):
        return next_step
    
    # elif check if the ghost is chasing us, if not, return no control, or else escape
    elif(x_samesign and y_samesign):
        if(x_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 1))):
            next_step.append(1)
        elif(x_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 0))):
            next_step.append(0)
        if(y_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 3))):
            next_step.append(3)
        elif(y_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 2))):
            next_step.append(2)            
        return next_step
    else:
        next_step.append(4)
        return next_step

