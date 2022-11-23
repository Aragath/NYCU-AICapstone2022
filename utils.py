from meta import GameMeta
import numpy as np
import random

# ==============================================================================
# ============================ to be deleted ===================================
# ==============================================================================
def get_random_board():
    board = np.ones((GameMeta.BOARD_SIZE, GameMeta.BOARD_SIZE), dtype=int)
    board = -board
    frees = {(
        np.random.randint(low=0, high=GameMeta.BOARD_SIZE),
        np.random.randint(low=0, high=GameMeta.BOARD_SIZE)
    )}
    
    while len(frees) < 64:
        src = random.choice(list(frees))
        dir = random.choice(GameMeta.DIRECTIONS)
        pattern = get_direction_pattern(src, dir)
        x = src[0] + pattern[0]
        y = src[1] + pattern[1]

        if not out_of_bound((x,y)):
            board[x][y] = GameMeta.TOKENS["free"]
            frees.add((x,y))
    
    return board

def print_state(mapState, sheepState):
    for i in range(GameMeta.BOARD_SIZE):
        if i % 2 == 1:
            print(" ", end="")
        for j in range(GameMeta.BOARD_SIZE):
            x = mapState[i][j]
            token = ( int(x) if x != GameMeta.TOKENS["wall"] else "x" )
            print(f"{token}", end=" ")
            
        if i % 2 == 0:
            print("  || ", end="")
        else:
            print(" ||  ", end="")

        for j in range(GameMeta.BOARD_SIZE):
            x = sheepState[i][j]
            token = ( int(x) if x != 0 else "x" )
            print(f"{token}", end=" ")
            
        print("")

def print_board(board):
    for i, row in enumerate(board):
        if i % 2 == 1:
            print(" ", end="")
        for x in row:
            token = ( int(x) if x != GameMeta.TOKENS["wall"] else "x" )
            print(f"{token} ", end="")
        print("")

# ==============================================================================
# ==============================================================================

def get_direction_pattern(cell, dir) -> tuple:
    x, _ = cell

    if dir == 1:
        return ((-1, -1) if x % 2 == 0 else (-1, 0))
    elif dir == 2:
        return ((-1, 0) if x % 2 == 0 else (-1, 1))
    elif dir == 3:
        return (0, -1)
    elif dir == 4:
        return (0, 1)
    elif dir == 5:
        return ((1, -1) if x % 2 == 0 else (1, 0))
    elif dir == 6:
        return ((1, 0) if x % 2 == 0 else (1, 1))
    # ============= for debug purposes, delete before submission =============
    else:
        print(f"Unrecognized direction: {dir}.")
        raise ValueError()
    # ========================================================================

def out_of_bound(cell: tuple) -> bool:
    x, y = cell

    if x < 0 or x >= GameMeta.BOARD_SIZE:
        return True
    elif y < 0 or y >= GameMeta.BOARD_SIZE:
        return True
    else:
        return False

def evaluate_init_pos(pos, board):
    i, j = pos
    if board[i][j] != GameMeta.TOKENS["free"]:
        return -GameMeta.INF

    frees = 0  # sum of distance can go in all 6 directions
    walls = 0  # number of walls in adjacent cells

    for d in GameMeta.DIRECTIONS:
        dest = get_dest(pos, d, board) # destination
        dist = distance(src=pos, dst=dest, dir=d)
        neighbor = get_neighbor(dest, d, board) # if going 1 step further from destination

        # ============= for debug purposes, delete before submission =============
        assert neighbor != GameMeta.TOKENS["free"]
        # ========================================================================
        
        if neighbor == GameMeta.TOKENS["wall"]:
            frees += dist

            if dist == 0:
                walls += 1
        else: # there is an oppenent
            frees += dist / 2 
    
    # at least 1 neighbor has to be wall
    return ( frees if walls != 0 else -GameMeta.INF )

def get_dest(src, dir, board) -> tuple:
    dest_x, dest_y = src
        
    while True:
        pattern_x, pattern_y = get_direction_pattern((dest_x, dest_y), dir)
        next_x = dest_x + pattern_x
        next_y = dest_y + pattern_y

        if out_of_bound((next_x, next_y)) or board[next_x][next_y] != GameMeta.TOKENS["free"]:
            break
            
        dest_x = next_x
        dest_y = next_y

    return (dest_x, dest_y)

def get_neighbor(pos, dir, board):
    x, y = pos
    pat_x, pat_y = get_direction_pattern(pos, dir)
    n_x, n_y = x+pat_x, y+pat_y

    if out_of_bound((n_x, n_y)):
        return GameMeta.TOKENS["wall"]
    else:
        return board[n_x][n_y]

def distance(src, dst, dir):
    return ( abs(src[1]-dst[1]) if dir in [3, 4] else abs(src[0]-dst[0]) )
    
# find sources and destinations of moves between 1 round
# at most GameMeta.PLAYER - 1 moves
def find_src_dst(prevState, mapState: list, sheepState: list):
    sources = {}
    destinations = {}

    for i in range(GameMeta.BOARD_SIZE):
        for j in range(GameMeta.BOARD_SIZE):
            # source
            if prevState.sheepState[i][j] > sheepState[i][j]:
                # ============= for debug purposes, delete before submission =============
                assert prevState.mapState[i][j] == mapState[i][j]
                assert (i, j) in prevState.cell_owners[mapState[i][j]]["free"]
                # assert mapState[i][j] != prevState.currentPlayer
                # ========================================================================
                sources[mapState[i][j]] = (i, j)

            # destination
            elif prevState.sheepState[i][j] < sheepState[i][j]:
                # ============= for debug purposes, delete before submission =============
                try:
                    assert prevState.mapState[i][j] == GameMeta.TOKENS["free"]
                    assert prevState.sheepState[i][j] == 0
                    assert (i, j) not in prevState.cell_owners[mapState[i][j]]["free"]
                except AssertionError:
                    print((i, j))
                    print_board(mapState)
                    print("\n")
                    print_board(sheepState)
                    raise AssertionError


                # ========================================================================
                destinations[mapState[i][j]] = (i, j)

            # ============= maybe check players with "free" only? =============
            # if len(sources) == GameMeta.PLAYERS - 1 and len(destinations) == GameMeta.PLAYERS - 1:
            #     break

    return sources, destinations