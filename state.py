from meta import GameMeta
from utils import get_direction_pattern, out_of_bound, get_neighbor
from group import Group

class Move:
    def __init__(self, cell: tuple, sheep=None, dir=None, init=False, stay=False):
        self.cell = cell    # position (y, x)
        self.sheep = sheep  # #sheep moving to new place
        self.dir = dir      # 1~6
        self.init = init    # the move sets the 1st piece for the player
        self.stay = stay    # True if not moving at all
        
        if init:
            assert not self.sheep and not self.dir and not self.stay

    def get_step(self):
        if self.init:
            return list(self.cell)
        else:
            return [list(self.cell), self.sheep, self.dir]

    def print_move(self):
        print(self.hash)

    # return the hash value for "children" in mcts/Node"
    @property
    def hash(self):
        if self.init:
            return f"init {self.cell}"
        elif self.stay:
            return "stay"
        else:
            return f"src: {self.cell}, sheep: {str(int(self.sheep))}, direction: {self.dir}"

    @staticmethod
    def src_dst_to_move(src: tuple, dst: tuple, sheep):
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        d = 0

        if dx < 0:
            if dy < 0:
                d = 1
            elif dy > 0:
                d = 2
            else:
                d = (2 if src[0] % 2 == 0 else 1)
        elif dx > 0:
            if dy < 0:
                d = 5
            elif dy > 0:
                d = 6
            else:
                d = (6 if src[0] % 2 == 0 else 5)
        else:
            if dy < 0:
                d = 3
            elif dy > 0:
                d = 4
            else:
                raise ValueError("Invalid src_dst")

        return Move(
            cell=src, 
            sheep=sheep,
            dir=d
        )

    # return a Move which indicates not moving
    @staticmethod
    def freeze():
        return Move(cell=None, stay=True)

class State:
    FREE = "free"
    BLOCKED = "blocked"

    def __init__(self, mapState: list, sheepState: list, current_player):
        self.mapState = mapState
        self.sheepState = sheepState
        self.current_player = current_player # 1 ~ GameMeta.PLAYERS
        self._init_cell_owners()       
        # keep track of cells of each player
        # { 
        #   1: { blocked: {}, free: {} }, token1
        #   2: { blocked: {}, free: {} }, token2
        #               ...
        # }

    def print_state(self):
        for i in range(GameMeta.BOARD_SIZE):
            if i % 2 == 1:
                print(" ", end="")
            for j in range(GameMeta.BOARD_SIZE):
                x = self.mapState[i][j]
                token = ( int(x) if x != GameMeta.TOKENS["wall"] else "x" )
                print(f"{token}", end=" ")
            
            if i % 2 == 0:
                print("  || ", end="")
            else:
                print(" ||  ", end="")

            for j in range(GameMeta.BOARD_SIZE):
                x = self.sheepState[i][j]
                token = ( int(x) if x != 0 else "x" )
                print(f"{token}", end=" ")
            
            print("")

    # find the destination of a Move
    def get_dest(self, move: Move) -> tuple:
        dest_x, dest_y = move.cell
        
        while True:
            pattern_x, pattern_y = get_direction_pattern((dest_x, dest_y), move.dir)
            next_x = dest_x + pattern_x
            next_y = dest_y + pattern_y

            if out_of_bound((next_x, next_y)) or self.mapState[next_x][next_y] != GameMeta.TOKENS["free"]:
                break
            
            dest_x = next_x
            dest_y = next_y
        
        # ============= for debug purposes, delete before submission ============= 
        
        if move.cell[0] == dest_x and move.cell[1] == dest_y:
            # The given Move does not move at all
            raise ValueError("This is an invalid move!")

        # ========================================================================

        return (dest_x, dest_y)
    
    @staticmethod
    def get_next_player(player): 
        return player % GameMeta.PLAYERS + 1 # 1~GameMeta.PLAYERS

    # return the number of plays of "player" so far
    def get_play_cnt(self, player):
        cells = self.cell_owners[player]
        return len(cells[self.BLOCKED]) + len(cells[self.FREE])

    def play(self, move: Move):
        if move.init:
            self.init_pos(move)
            return

        # not moving at all
        if move.stay:
            self.current_player = self.get_next_player(self.current_player)
            return

        dest_x, dest_y = self.get_dest(move)
        
        # ============= for debug purposes, delete before submission ============= 
        
        if self.mapState[move.cell[0]][move.cell[1]] != self.current_player:
            print(f"{(move.cell)}={self.mapState[move.cell[0]][move.cell[1]] } does not belong to the current player {self.current_player}.")
            print(self.cell_owners)
            raise ValueError()

        if self.mapState[dest_x][dest_y] != GameMeta.TOKENS["free"]:
            raise ValueError("The destination cell is unavailable.")

        if self.sheepState[move.cell[0]][move.cell[1]] <= move.sheep:
            print(self.current_player)
            print(self.cell_owners)
            raise ValueError(f"{move.cell} does not have enough sheep.")

        if self.sheepState[dest_x][dest_y] != 0:
            raise ValueError("The destination cell is already occupied by sheep.")

        # ========================================================================

        self.sheepState[move.cell[0]][move.cell[1]] -= move.sheep

        self.mapState[dest_x][dest_y] = self.current_player
        self.sheepState[dest_x][dest_y] = move.sheep
        self.cell_owners[self.current_player][self.FREE].add((dest_x, dest_y))

        self.current_player = self.get_next_player(self.current_player)

    def _init_cell_owners(self):
        self.cell_owners = {}
        for i in range(1, GameMeta.PLAYERS+1):
            self.cell_owners[i] = {
                self.FREE: set(),
                self.BLOCKED: set(),
            }

        for i in range(GameMeta.BOARD_SIZE):
            for j in range(GameMeta.BOARD_SIZE):
                player = self.mapState[i][j]
                if player in self.cell_owners:
                    self.cell_owners[player][self.FREE].add((i, j))

    # ************************************************************************
    # ************************** for move selections *************************
    # ************************************************************************

    def get_valid_moves(self) -> list:
        cells = self.cell_owners[self.current_player]
        if len(cells[self.FREE]) + len(cells[self.BLOCKED]) == 0:
            return self._get_valid_init_moves()

        valid_move_list = []        
        blocked_cells = []
        for cell in self.cell_owners[self.current_player][self.FREE]:
            valid_directions = self._get_valid_directions(cell)
            valid_sheep = self._get_valid_sheep(cell)

            if len(valid_directions) == 0 or len(valid_sheep) == 0:
                blocked_cells.append(cell)
                continue

            for direction in valid_directions:
                for num_sheep in valid_sheep:
                    valid_move_list.append(Move(cell, num_sheep, direction))
        
        for cell in blocked_cells:
            self.cell_owners[self.current_player][self.FREE].remove(cell)
            self.cell_owners[self.current_player][self.BLOCKED].add(cell)

        # cannot move at all
        if len(valid_move_list) == 0:
            valid_move_list.append(Move.freeze())

        return valid_move_list

    def init_pos(self, move: Move):
        x, y = move.cell

        # ============= for debug purposes, delete before submission ============= 
        
        if self.mapState[x][y] != GameMeta.TOKENS["free"] or self.sheepState[x][y] != 0:
            print(f"{move.cell} is blocked")
            self.print_state()
            raise ValueError()

        if len(self.cell_owners[self.current_player][self.FREE]) != 0 or len(self.cell_owners[self.current_player][self.BLOCKED]) != 0:
            print("The player has init_pos already")
            raise ValueError()
        
        # ========================================================================

        self.mapState[x][y] = self.current_player
        self.sheepState[x][y] = GameMeta.MAX_SHEEP
        self.cell_owners[self.current_player][self.FREE].add(move.cell)

        self.current_player = self.get_next_player(self.current_player)

    def _get_valid_init_moves(self) -> list:
        valid_moves = []
        
        for i in range(GameMeta.BOARD_SIZE):
            for j in range(GameMeta.BOARD_SIZE):
                if self.mapState[i][j] != GameMeta.TOKENS["free"]:
                    continue

                walls = 0
                frees = 0
                for d in GameMeta.DIRECTIONS:
                    neighbor = get_neighbor((i, j), d, self.mapState)
                    if neighbor == GameMeta.TOKENS["wall"]:
                        walls += 1
                    elif neighbor == GameMeta.TOKENS["free"]:
                        frees += 1

                    if walls and frees:
                        break

                # 1. a valid init pos must be adjacent to at least 1 wall
                # 2. a cell can be valid even if blocked in all 6 directions
                #    but it makes no sense to init at such a position
                if walls and frees:
                    valid_moves.append(Move(cell=(i,j), init=True))

        return valid_moves

    # return valid direction to move for the given cell
    def _get_valid_directions(self, cell: tuple) -> list:
        valid_directions = []

        for dir in GameMeta.DIRECTIONS:
            pattern = get_direction_pattern(cell, dir)
            next_cell = (cell[0] + pattern[0], cell[1] + pattern[1])

            if not out_of_bound(next_cell) and self.mapState[next_cell[0]][next_cell[1]] == GameMeta.TOKENS["free"]:
                valid_directions.append(dir)

        return valid_directions

    # return valid number of sheep to move for the given cell
    def _get_valid_sheep(self, cell: tuple) -> list:
        x, y = cell

        min_sheep = 1
        max_sheep = int(self.sheepState[x][y])-1

        return list(range(min_sheep, max_sheep+1))

    # ************************************************************************
    # ************************ for score calculations ************************
    # ************************************************************************

    def gameover(self) -> bool:
        blocked_cells = 0

        for cells in self.cell_owners.values():
            if len(cells[self.FREE]) != 0:
                return False
            blocked_cells += len(cells[self.BLOCKED])
        
        if blocked_cells == 0: # the game has not even started
            return False
        else:
            return True

    # return points of all players based on rankings & scores
    def get_points(self) -> dict:
        scores = [ (self._get_score(p), p) for p in self.cell_owners ] # (score, player)
        scores = sorted(scores, reverse=True)
        
        rankings = [[scores[0][1]]]
        for i in range(1, len(scores)):
            if scores[i][0] == scores[i-1][0]: # score ties
                rankings[-1].append(scores[i][1])
                rankings.append([])
            else:
                rankings.append([scores[i][1]])

        points = {}
        for ranking, players in enumerate(rankings):
            # REWARDS index starts from 0
            for player in players:
                points[player] = GameMeta.REWARDS[ranking] / len(players)

        return points

    # return scores of a player
    def _get_score(self, player):
        # ============= for debug purposes, delete before submission =============
        if len(self.cell_owners[player][self.FREE]) != 0:
            print("The game is not over.")
            raise ValueError()
        # ========================================================================
        
        cells = self.cell_owners[player]["blocked"]
        groups = Group(cells).get_connect_regions()
        
        score = 0
        max_size = 0
        for group in groups:
            score += 3 * len(group)

            if len(group) > max_size:
                max_size = len(group)

        return score+max_size
