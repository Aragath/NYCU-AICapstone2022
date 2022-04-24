import random
import threading
import python.STcpClient as STcpClient
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

def hit_the_wall(x_currentnode, y_currentnode, move_index):      #the move_index here is not as same as the action, just an index
  global parallel_wall, vertical_wall
  col_index = (x_currentnode-1)//25
  row_index = (y_currentnode-1)//25

  if(move_index == 0): #move up, check the parallel_wall 
    if(parallel_wall[col_index][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 1): #move down, check the parallel_wall 
    if(parallel_wall[col_index][row_index+1] == 1):
      return True
    else:
      return False
  elif(move_index == 2): #move left, check the vertical_wall
    if(vertical_wall[col_index][row_index] == 1):
      return True
    else:
      return False
  elif(move_index == 3): #move right, check the vertical_wall 
    if(vertical_wall[col_index+1][row_index] == 1):
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
      for index, node in emuerate(white_list):
          if node.f < gray_node.f:
              gray_index = index
              gray_node = node

      white_list.pop(gray_index)
      black_list.append(gray_node)

      #found the goal_node
      if gray_node == goal_node:
          path = []
          current = gray_node
          while current is not None
              path.append(current.position)
              current = current.parent
          next_pos =  path[len(path)-2]          #the next position which the agent needs to go to
          return what_is_next(start_pos, next_pos) 
      #not yet found the goal_node
      children = []
      for move_index, move_to in enumerate([(0, -5), (0, 5), (-5, 0), (5, 0)]):    #5 points per time step
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
    '''
    control of your player
    0: left, 1:right, 2: up, 3: down 4:no control
    format is (control, set landmine or not) = (0~3, True or False)
    put your control in action and time limit is 0.04sec for one step
    '''

    start_pos = (playerStat[0], playerStat[1])
    goal_pos = (propsStat[0][1], propsStat[0][2])     #can be changed

    #move = random.choice([2])
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
        is_connect=STcpClient.SendStep(id_package, action[0], action[1])
        if not is_connect:
            break
