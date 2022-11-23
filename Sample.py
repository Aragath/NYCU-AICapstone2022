import STcpClient
from meta import GameMeta
import numpy as np
from state import State, Move
from mcts import MCTS
from utils import *

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''

# def InitPos(mapStat):
#     best_pos = [0, 0]
#     max_score = -GameMeta.INF
#     mapStat = np.transpose(np.array(mapStat))

#     for i in range(GameMeta.BOARD_SIZE):
#         for j in range(GameMeta.BOARD_SIZE):
#             score = evaluate_init_pos((i, j), mapStat)

#             if score > max_score:
#                 best_pos = [i, j]
#                 max_score = score
    
#     # mapStat is column major
#     # but self-defined functions are row major
#     best_pos = [best_pos[1], best_pos[0]]
#     return best_pos

def InitPos(mapStat):
    best_pos = [0, 0]
    max_score = -GameMeta.INF
    opponents_pos = set()

    for i in range(GameMeta.BOARD_SIZE):
        for j in range(GameMeta.BOARD_SIZE):
            if mapStat[i][j] > 0:
                opponents_pos.add((i, j))
            score = evaluate_init_pos((i, j), mapStat)

            if score > max_score:
                best_pos = [i, j]
                max_score = score
    
    # this is the last player (player 4) for InitPos
    # who actually can start running MCTS befre GetStep
    if len(opponents_pos) == GameMeta.PLAYERS-1:
        playerID = GameMeta.PLAYERS
        sheepState = np.zeros((GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE), dtype=int)
        for pos in opponents_pos:
            sheepState[pos[0]][pos[1]] = GameMeta.MAX_SHEEP

        state = State(mapState=mapStat, sheepState=sheepState, current_player=playerID)
        state.init_pos(tuple(best_pos))
        
        global agent
        agent = MCTS(state)
        agent.search(time_budget=GameMeta.TIME_LIMIT*0.9)

    best_pos = [best_pos[1], best_pos[0]]
    return best_pos




'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~6),對應方向如下圖所示
              1  2
            3  x  4
              5  6
'''

def GetStep(playerID, mapStat, sheepStat):
    # step = [(0, 0), 0, 1]
    
    # Write your code here
    global agent
    mapStat = np.transpose(np.array(mapStat))
    sheepStat = np.transpose(np.array(sheepStat))

    # update the state
    if agent is None: # 1st call after InitPos
        state = State(mapState=mapStat, sheepState=sheepStat, current_player=playerID)
        agent = MCTS(state)
    else:
        sources, destinations = find_src_dst(agent.root_state, mapStat, sheepStat)
    
        currentPlyer = playerID
        for _ in range(GameMeta.PLAYERS-1):
            currentPlyer = State.get_next_player(currentPlyer)
            if currentPlyer == playerID:
                break
            else:
                # the player did not move
                if currentPlyer not in sources:
                    # ============= for debug purposes, delete before submission =============
                    assert currentPlyer not in destinations
                    # ========================================================================
                    agent.move_to(Move.freeze(), msg="update")
                else:
                    src = sources[currentPlyer]
                    dst = destinations[currentPlyer]    
                    move = Move.src_dst_to_move(src, dst, sheepStat[dst[0]][dst[1]])
                    agent.move_to(move, msg="update")

    assert (agent.root_state.mapState == mapStat).all()
    assert (agent.root_state.sheepState == sheepStat).all()

    agent.search(time_budget=GameMeta.TIME_LIMIT * 0.95)
    best_move = agent.get_best_move()
    agent.move_to(best_move, "best move")

    agent.root_state.print_state()

    # mapStat is column major
    # but self-defined functions are row major
    step = best_move.get_step()
    step[0][0], step[0][1] = step[0][1], step[0][0]
    return step


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

agent = None

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
