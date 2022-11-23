class MCTSMeta:
    EXPLORE_WEIGHT = 2.0 # for ucb exploration
    H_WEIGHT = 0       # for heuristic
    H_DECAY = True       # gradually reduce weight of heuristic as game goes on


class GameMeta:
    PLAYERS = 4
    INF = float('inf')

    REWARDS = [5, 3, 2, 1] # points for players with 1st, 2nd, 3rd, 4th highest points
    DIRECTIONS = [1, 2, 3, 4, 5, 6]
    TOKENS = {"wall": -1, "free": 0, "player1": 1, "player2": 2, "player3": 3, "player4": 4}
    
    TIME_LIMIT = 5.0
    BOARD_SIZE = 12
    MIN_SHEEP = 1
    MAX_SHEEP = 16