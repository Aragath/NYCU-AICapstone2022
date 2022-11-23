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
      if(mindis <= 50**2 + 50**2):     #very very very close
        break
   goal = (x, y)
   return goal


def what_is_next(start_pos, next_pos):            #to determine the direction (up, down, left, right)
   if next_pos[0] - start_pos[0] == 0:      #only y's dif
      dif_y = next_pos[1] - start_pos[1]
      if dif_y > 0:
          direction = 3       #down
      if dif_y < 0:
          direction = 2       #up
   if next_pos[1] - start_pos[1] == 0:      #only x's dif
      dif_x = next_pos[0] - start_pos[0]
      if dif_x > 0:
          direction = 1       #right
      if dif_x < 0:
          direction = 0       #left

   return direction

def hit_the_wall(x_currentnode, y_currentnode, move_index):
  global parallel_wall, vertical_wall
  col_index = (x_currentnode-1)//25
  row_index = (y_currentnode-1)//25

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

def A_Star_Search(start_pos, goal_pos):           #return next move

   start_node = AstarNode(None, start_pos)
   goal_node = AstarNode(None, goal_pos)
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
          next_pos =  path[len(path)-2]          #the next position which the agent needs to go to
          return what_is_next(start_pos, next_pos) 
      #not yet found the goal_node
      children = []
      for move_index, move_to in enumerate([(-5, 0), (5, 0), (0, -5), (0, 5)]):    #5 points per time step
          new_node_pos = (gray_node.position[0] + move_to[0], gray_node.position[1] + move_to[1])
          if new_node_pos[0] > 376 or new_node_pos[0] < 1 or new_node_pos[1] > 376 or new_node_pos[1] < 1:    #check the boarder
              continue
          if hit_the_wall(gray_node.position[0], gray_node.position[1], move_index) == True:
              continue
          new_node = AstarNode(gray_node, new_node_pos)
          children.append(new_node)
      for child in children:     #check whether the child is in the black list 
          for black_child in black_list:
              if child == black_child:
                  continue
          child.g = gray_node.g + 25
          child.h = ((child.position[0] - goal_node.position[0]) ** 2 + (child.position[1] - goal_node.position[1]) ** 2)   #heuristic
          child.f = child.g + child.h

          for white_node in white_list:   #check whether child is already in the white list and this node can't renew the old node
              if child == white_node and child.g > white_node.g:
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
            print("chase the ghost near, go left\n")
        elif(x_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 1))):
            next_step.append(1)
            print("chase the ghost near, go right\n")
        if(y_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 2))):
            next_step.append(2)
            print("chase the ghost near, go up\n")
        elif(y_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 3))):
            next_step.append(3)            
            print("chase the ghost near, go down\n")
        return next_step        
    
    # if no ghost is near, return empty list
    if(dis_between * 25 < nearest_dis):
        print("no ghost near\n")
        return next_step
    # elif check if the ghost is chasing us, if yes, escape
    elif(x_samesign and y_samesign):
        if(x_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 1))):
            next_step.append(1)
            print("escape ghost near, go right\n")
        elif(x_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 0))):
            next_step.append(0)
            print("escape ghost near, go left\n")
        if(y_between > 0 and (not hit_the_wall(playerState[0], playerState[1], 3))):
            next_step.append(3)
            print("escape ghost near, go down\n")
        elif(y_between < 0 and (not hit_the_wall(playerState[0], playerState[1], 2))):
            next_step.append(2)
            print("escape ghost near, go up\n")

        if(len(next_step) == 0):      # trigger escape but all escaping direction hit the wall
            if(x_between == 0):           # final escaping method: turn left or right
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
    global planB
    '''
    control of your player
    0: left, 1:right, 2: up, 3: down 4:no control
    format is (control, set landmine or not) = (0~3, True or False)
    put your control in action and time limit is 0.04sec for one step
    '''
    planB = random.choice([0, 1, 2, 3])

    dis_between = 4     # if the nearest ghost is smaller than [dis_between], escape
    dis_to_chase = 6    # if ghost in [dis_to_chase] blocks, chase it

    start = time.time()
    urgencylist = EatOrRun(playerStat, ghostStat)
    end = time.time()
    print("time eatorrun cost: ", end - start, "\n")
    start_pos = (playerStat[0], playerStat[1])

    if(len(urgencylist) == 1):
        move = urgencylist[0]           #do this first, don't consider the pellet
        planB = move
    elif(playerStat[3] > 0):
        move = urgencylist[0]           # two direction possible and under supermode, so do urgency first
        planB = move
    else:
        start2 = time.time()
        goal_pos = find_one_goal_pellet(start_pos, propsStat)
        end2 = time.time()
        print("time find_one_goal_pellet cost: ", end2 - start2, "\n")
        # don't do Astar if goal is really close
        if(abs(start_pos[0]- goal_pos[0]) + abs(start_pos[0]- goal_pos[0]) <= 2*25):
            print("doing trivial case\n")
            candidate = []
            x_between = start_pos[0] - goal_pos[0]
            y_between = start_pos[1] - goal_pos[1]
            print("x between: ", x_between)
            print("y between: ", y_between)
            if(x_between >= 15 and (not hit_the_wall(start_pos[0], start_pos[1], 1))):
                print("cand is 1")
                candidate.append(1)
            elif(x_between < -10 and (not hit_the_wall(start_pos[0], start_pos[1], 0))):
                print("cand is 0")
                candidate.append(0)
            elif(y_between > 0 and (not hit_the_wall(start_pos[0], start_pos[1], 2))):
                print("cand is 2")
                candidate.append(2)
            elif(y_between < 0 and (not hit_the_wall(start_pos[0], start_pos[1], 3))):
                print("cand is 3")
                candidate.append(3)
            if(candidate == []):
                move = random.choice([0, 1, 2, 3])
            else:
                move = random.choice(candidate)
        # do Astar when goal is far
        else:
            print("doing A_Star_Search\n")
            start = time.time()
            direction = A_Star_Search(start_pos, goal_pos)
            end = time.time()
            print("time A_Star_Search cost: ", end - start, "\n")
            print("move changed, and this is the A star search", direction)
            if(len(urgencylist) == 0):
                move = direction                #just consider the pellet
                planB = move
            elif(len(urgencylist) == 2):
                move = hardship(urgencylist, direction)     #see the function description to get the details
                planB = move

    if playerStat[2] > 0:
        landmine = random.choice([True, False])
    else:
        landmine = False
    print("this is move: ", move)
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

    while True:
        # playerStat: [x, y, n_landmine,super_time, score]
        # otherplayerStat: [x, y, n_landmine, super_time]
        # ghostStat: [[x, y],[x, y],[x, y],[x, y]]
        # propsStat: [[type, x, y] * N]
        (stop_program, id_package, playerStat,otherPlayerStat, ghostStat, propsStat) = STcpClient.GetGameStat()
        print("this is playerPos", playerStat[0], playerStat[1])
        if stop_program:
            break
        elif stop_program is None:
            break
        global action
        action = None
        user_thread = MyThread(target=getStep, args=(playerStat, ghostStat, propsStat))
        user_thread.start()
        time.sleep(4/100)
        #time.sleep(4)

        if action == None:
            user_thread.kill()
            user_thread.join()
            #action = [4, False]
            action = [planB, False]
            print("Over time, use planB", planB)
        is_connect=STcpClient.SendStep(id_package, action[0], action[1])
        if not is_connect:
            break

        prev_pos = ghostStat