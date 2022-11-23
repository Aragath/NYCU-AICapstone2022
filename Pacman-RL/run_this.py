import random
import threading
import python.STcpClient as STcpClient
import time
import sys

from board_env import Board


action_list = [[0, False], [1, False], [2, False], [3, False], [4, False]]    #testing the simplest in advance, otherwise the q_table will be larger
#action_list = [[0, False], [0, True], [1, False], [1, True], [2, False], [2, True], [3, False], [3, True], [4, False], [4, True]]

class QTable:
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

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

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        while True:
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learn from this transition
            RL.learn(observation, action, reward, observation_)
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
    # end of game
    print('game over')
    env.clear()


def getStep(playerStat, ghostStat, propsStat):
    global action
    '''
    control of your player
    0: left, 1:right, 2: up, 3: down 4:no control
    format is (control, set landmine or not) = (0~3, True or False)
    put your control in action and time limit is 0.04sec for one step
    '''
    RL = QTable(actions = action_list)
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

    # initial event                                                                #added
    env = Board()
    # initial Q table
    RL = QTable(actions = action_list)                        #added

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
