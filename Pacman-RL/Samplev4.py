import random
import threading
import STcpClient as STcpClient
import time
import sys

class AstarNode():
   def __init__(self, parent = None, position = None):
       self.parent = parent
       self.position = position
       self.f = 0               #f()
       self.g = 0               #g()
       self.h = 0               #h()
   def __eq__(self, other):
       return self.position == other.position
   def compare(self, other):
       if(abs(self.position[0] - other.position[0]) <= 10 and abs(self.position[1] - other.position[1]) <= 10):
          return True
       else:
          return False


def find_one_goal_pellet(start_pos, propsStat):
   n_props = len(propsStat)
   mindis = 400**2 + 400**2
   for i in range(n_props):
      if(propsStat[i][0] != 2):
        continue
      dis = (propsStat[i][1] - start_pos[0])**2 + (propsStat[i][2] - start_pos[1])**2
      if(dis < mindis):
        mindis = dis
        x = propsStat[i][1]
        y = propsStat[i][2]
   goal = (x, y)
   print("goal pellet is: \n", transform(goal))
   return goal


def what_is_next(start_pos, next_pos):            #to determine the direction (up, down, left, right)

   d = 4
   if next_pos[0] - start_pos[0] == 0:      #only y's dif
      dif_y = next_pos[1] - start_pos[1]
      if dif_y > 0:
          d = 3       #down
      if dif_y < 0:
          d = 2       #up
   if next_pos[1] - start_pos[1] == 0:      #only x's dif
      dif_x = next_pos[0] - start_pos[0]
      if dif_x > 0:
          d = 1       #right
      if dif_x < 0:
          d = 0       #left

   return d

def hit_the_wall(x_currentnode, y_currentnode, move_index):
  global parallel_wall, vertical_wall
  col_index = x_currentnode//25
  row_index = y_currentnode//25

  if(move_index == 0): #move left, check the vertical_wall
    if(vertical_wall[col_index][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 1): #move right, check the vertical_wall
    if(vertical_wall[col_index+1][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 2): #move up, check the parallel_wall
    if(parallel_wall[col_index][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 3): #move down, check the parallel_wall
    if(parallel_wall[col_index][row_index+1] == 1):
      return True
    else:
      return False

def hit_the_wall_v2(x_currentnode, y_currentnode, move_index):
  global parallel_wall, vertical_wall
  col_index = x_currentnode
  row_index = y_currentnode

  if(move_index == 0): #move left, check the vertical_wall
    if(vertical_wall[col_index][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 1): #move right, check the vertical_wall
    if(vertical_wall[col_index+1][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 2): #move up, check the parallel_wall
    if(parallel_wall[col_index][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 3): #move down, check the parallel_wall
    if(parallel_wall[col_index][row_index+1] == 1):
      return True
    else:
      return False

def transform(pos):
   x_new = pos[0]//25
   y_new = pos[1]//25
   new_pos = (x_new, y_new)

   return new_pos

def A_Star_Search(start_pos, goal_pos):           #return next move

   start_pos_transformed = transform(start_pos)
   goal_pos_transformed = transform(goal_pos)
   start_node = AstarNode(None, start_pos_transformed)
   goal_node = AstarNode(None, goal_pos_transformed)
   start_node.f = start_node.g = start_node.h = 0
   goal_node.f = goal_node.g = goal_node.h = 0

   white_list = []      #white means: the node hasn't been explored
   black_list = []      #black means: the node has been explored
   white_list.append(start_node)

   while len(white_list) > 0:
      gray_node = white_list[0]      #gray means: we are currently exploring this node
      gray_index = 0
      for index, node in enumerate(white_list):
          if node.f < gray_node.f:
              gray_index = index
              gray_node = node

      white_list.pop(gray_index)
      black_list.append(gray_node)

      #found the goal_node
      if gray_node == goal_node:
          path = []
          current = gray_node
          while current is not None:
              path.append(current.position)
              current = current.parent
          if(len(path) > 1):
            next_pos =  path[-2]          #the next position which the agent needs to go to
          elif(len(path) == 1):           #I don't know why I should add this condition
            next_pos = path[0]
          print(path[::-1])
          go_to = what_is_next(start_pos_transformed, next_pos)
          print("go_to: ", go_to)
          return go_to
      #not yet found the goal_node
      children = []
      for move_index, move_to in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):    #5 points per time step
          new_node_pos = (gray_node.position[0] + move_to[0], gray_node.position[1] + move_to[1])
          if new_node_pos[0] > 15 or new_node_pos[0] < 0 or new_node_pos[1] > 15 or new_node_pos[1] < 0:    #check the boarder
              continue
          if hit_the_wall_v2(gray_node.position[0], gray_node.position[1], move_index) == True:
              continue
          new_node = AstarNode(gray_node, new_node_pos)
          children.append(new_node)
      for child in children:     #check whether the child is in the black list 
          for black_child in black_list:
              if child == black_child:
                  continue
          child.g = gray_node.g + 1
          child.h = (abs(child.position[0] - goal_node.position[0]) + abs(child.position[1] - goal_node.position[1]))   #heuristic
          child.f = child.g + child.h

          for white_node in white_list:   #check whether child is already in the white list and this node can't renew the old node
              if child == white_node  and child.g > white_node.g:
                  continue
          white_list.append(child)

def EatOrRun(playerState, ghostState) -> list:
    global prev_pos, dis_between, dis_to_chase
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
    # elif check if the ghost is chasing us, if yes, escape
    elif(x_samesign and y_samesign):
        if(x_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 1))):
            next_step.append(1)
        elif(x_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 0))):
            next_step.append(0)
        if(y_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 3))):
            next_step.append(3)
        elif(y_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 2))):
            next_step.append(2)

        if(len(next_step) == 0):      #all escaping direction hit the wall
            if(x_between == 0):           #final escaping method: turn left or right
                if(not hit_the_wall(playerState[0], playerState[1], 0)):
                    next_step.append(0)
                if(not hit_the_wall(playerState[0], playerState[1], 1)):
                    next_step.append(1)
            elif(y_between == 0):         #final escaping method: turn up or down. https://imgur.com/aQyA7kv
                if(not hit_the_wall(playerState[0], playerState[1], 2)):
                    next_step.append(2)
                if(not hit_the_wall(playerState[0], playerState[1], 3)):
                    next_step.append(3)
            else:           #we will get closer to the ghost, but it's better than stand still. https://imgur.com/GjpRvHb
                if(not hit_the_wall(playerState[0], playerState[1], 0)):
                    next_step.append(0)
                if(not hit_the_wall(playerState[0], playerState[1], 1)):
                    next_step.append(1)
                if(not hit_the_wall(playerState[0], playerState[1], 2)):
                    next_step.append(2)
                if(not hit_the_wall(playerState[0], playerState[1], 3)):
                    next_step.append(3)
        return next_step
    #the ghost is near but not chasing us, don't wory
    else:
        return next_step

def hardship(urgencylist, direction):     #deal with the problem like: urglist tells you to turn left or down, but the AStatSearch tells you to turn up
   #0: left, 1:right, 2: up, 3: down
   result = direction
   dir1 = urgencylist[0]
   dir2 = urgencylist[1]
   if(direction != dir1 and direction != dir2):
      if((abs(direction - dir1) == 1) and (direction * dir1) != 2):
          result = dir2
      elif((abs(direction - dir2) == 1) and (direction * dir2) != 2):
          result = dir1
      else:
          result = random.choice([dir1, dir2])      #see the report to get the description. such like: https://imgur.com/aQyA7kv
   return result


class MyThread(threading.Thread): 
   def __init__(self, *args, **keywords): 
       threading.Thread.__init__(self, *args, **keywords) 
       self.killed = False      
   def start(self):         
       self.__run_backup = self.run         
       self.run = self.__run                
       threading.Thread.start(self)         
   def __run(self):         
       sys.settrace(self.globaltrace)         
       self.__run_backup()         
       self.run = self.__run_backup         
   def globaltrace(self, frame, event, arg):         
       if event == 'call':             
           return self.localtrace         
       else:             
           return None        
   def localtrace(self, frame, event, arg):         
       if self.killed:             
          if event == 'line':                 
              raise SystemExit()         
       return self.localtrace         
   def kill(self):         
       self.killed = True


def getStep(playerStat, ghostStat, propsStat):
    global action
    global parallel_wall, vertical_wall
    global prev_pos, dis_between, dis_to_chase
    '''
    control of your player
    0: left, 1:right, 2: up, 3: down 4:no control
    format is (control, set landmine or not) = (0~3, True or False)
    put your control in action and time limit is 0.04sec for one step
    '''

    start_pos = (playerStat[0], playerStat[1])
    goal_pos = find_one_goal_pellet(start_pos, propsStat)
    start = time.time()
    direction = A_Star_Search(start_pos, goal_pos)
    end = time.time()
    print("A Star Search cost: ", end - start, "\n")

    move = direction                #just consider the pellet

    landmine = False
    if playerStat[2] > 0:
      landmine = random.choice([True, False])
    action = [move, landmine]

    prev_pos = ghostStat.copy()
    #move = random.choice([0, 1, 2, 3, 4])
    #landmine = False
    #if playerStat[2] > 0:
    #    landmine = random.choice([True, False])
    #action = [move, landmine]


# props img size => pellet = 5*5, landmine = 11*11, bomb = 11*11
# player, ghost img size=23x23


if __name__ == "__main__":
    # parallel_wall = zeros([16, 17])
    # vertical_wall = zeros([17, 16])
    (stop_program, id_package, parallel_wall, vertical_wall) = STcpClient.GetMap()
    prev_pos = [[188, 188], [188, 188], [188, 188], [188, 188]]
    while True:
        # playerStat: [x, y, n_landmine,super_time, score]
        # otherplayerStat: [x, y, n_landmine, super_time]
        # ghostStat: [[x, y],[x, y],[x, y],[x, y]]
        # propsStat: [[type, x, y] * N]
        (stop_program, id_package, playerStat,otherPlayerStat, ghostStat, propsStat) = STcpClient.GetGameStat()
        if stop_program:
            break
        elif stop_program is None:
            break
        global action
        action = None
        user_thread = MyThread(target=getStep, args=(playerStat, ghostStat, propsStat))
        user_thread.start()
        time.sleep(4/100)
        if action == None:
            user_thread.kill()
            user_thread.join()
            action = [4, False]
            print("time out! \n")
            if(hit_the_wall(playerStat[0], playerStat[1], 0)):
              print("left will hit a wall")
            if(hit_the_wall(playerStat[0], playerStat[1], 1)):
              print("right will hit a wall")
            if(hit_the_wall(playerStat[0], playerStat[1], 2)):
              print("up will hit a wall")
            if(hit_the_wall(playerStat[0], playerStat[1], 3)):
              print("down will hit a wall")

        is_connect=STcpClient.SendStep(id_package, action[0], action[1])
        if not is_connect:
            break
