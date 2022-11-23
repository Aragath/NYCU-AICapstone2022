from math import sqrt, log
from copy import deepcopy
import random
from time import time as clock
from meta import GameMeta, MCTSMeta
from state import State, Move


class Node:
    def __init__(self, owner, explore_weight, h_weight, h_decay, move: Move = None, parent: object = None, plays=0):
        self.owner = owner # current player of the node
        self.move = move   # how to move from parent to here
        self.parent = parent
        self.plays = plays # number of plays of the player (owner) so far
        self.children = {}
        self.N = 0         # times this position was visited
        self.Q = 0         # total points of all leaves stemming from here
        self.H = None      # heuristics
        
        self.explore_weight = explore_weight
        self.h_weight = h_weight
        self.h_decay = h_decay

    def add_children(self, children: list):
        for child in children:
            assert State.get_next_player(self.owner) == child.owner
            self.children[child.move.hash] = child

    # self.H (the heuristic) = number of valid moves at the moment
    def set_H(self, state: State):
        assert state.current_player == self.owner

        if self.h_weight and not self.H:
            self.H = len(state.get_valid_moves())

    @property # calculate modified ucb value
    def value(self):
        # if not visited, set the value as infinity. 
        # Nodes with no visits are on priority
        if self.N == 0:
            return 0 if self.explore_weight == 0 else GameMeta.INF
        else:
            # exploitation + exploration
            ucb =  self.Q / self.N + self.explore_weight * sqrt(log(self.parent.N) / self.N)

            if self.h_weight == 0:
                return ucb
            elif not self.h_decay:
                return ucb + self.h_weight * self.H
            else:
                # rely less on heuristics as gane goes on
                return ucb + self.h_weight * self.H / sqrt(self.plays+1)  # +1 to avoid zero division error

class MCTS:
    def __init__(self, state: State, explore_weight=MCTSMeta.EXPLORE_WEIGHT, h_weight=MCTSMeta.H_WEIGHT, h_decay=MCTSMeta.H_DECAY):
        print("init mcts") # check if the global variable in test.py works
        
        self.explore_weight = explore_weight
        self.h_weight = h_weight
        self.h_decay = h_decay

        self.node_cnt = 0
        self.simulate_cnt = 0 # number of simulations
        self.reset_cnt = 0    # number of resests due to unreached state

        self.root_state = deepcopy(state) # probably no need to copy here?
        self.root = Node(
            **self._get_meta_pack(),
            owner=state.current_player, 
            plays=0,
        )
        

    def search(self, time_budget: float) -> None:
        start_time = clock()
        while clock() - start_time < time_budget:
            node, state = self.select() # the selected state is deepcopied
            self.simulate(state)        # random play until gameover
            self.backpropagate(node, state.get_points())
            self.simulate_cnt += 1
        
    def select(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        # stop if we find reach a leaf node
        while len(node.children) != 0:
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values() if n.value == max_value]
            node = random.choice(max_nodes)
            state.play(node.move)

            # if some child node has not been explored select it before expanding others
            if node.N == 0:
                return node, state

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.play(node.move)
        
        return node, state

    def expand(self, parent: Node, state: State) -> bool:
        if state.gameover():
            # game is over nothing to expand
            return False

        assert parent.owner == state.current_player

        children = []
        for move in state.get_valid_moves():
            if self.h_weight != 0:
                child_state = deepcopy(state)
                child_state.play(move)

                child = Node(
                    **self._get_meta_pack(),
                    owner=child_state.current_player,
                    plays=state.get_play_cnt(child_state.current_player),
                    move=move, 
                    parent=parent,
                )
                
                child.set_H(child_state) # set the heuristic value
            else:
                child = Node(
                    **self._get_meta_pack(),
                    owner=State.get_next_player(state.current_player),
                    move=move, 
                    parent=parent,
                )

            children.append(child)

        parent.add_children(children)
        self.node_cnt += len(children)

        return True

    @staticmethod
    def simulate(state: State):
        while not state.gameover():
            valid_moves = state.get_valid_moves()
            state.play(random.choice(valid_moves))

    @staticmethod
    def backpropagate(node: Node, points: list):
        # Caution! Q should += player who just played (parent.owner)
        # not who is going to play (node.owners)

        while node:
            node.N += 1
            if node.parent:
                node.Q += points[node.parent.owner]
            # ============= for debug purposes, delete before submission =============
            if node.parent:
                assert State.get_next_player(node.parent.owner) == node.owner
            # ========================================================================
            node = node.parent

    def get_best_move(self) -> Move:
        assert self.root_state.current_player == self.root.owner

        if self.root_state.gameover():
            return None

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        bestchild = random.choice(max_nodes)
        return bestchild.move

    def move_to(self, move: Move, msg: str=""):
        assert self.root_state.current_player == self.root.owner

        try:
            p_before_play = self.root_state.current_player

            child = self.root.children[move.hash]
            child.parent = None
            self.root = child
            self.root_state.play(child.move)

            if p_before_play == self.root_state.current_player:
                print("turn not changing")
                raise ValueError()

            return
            
        # it is possible that mcts has not reached "move"
        # in this case, reset "root_state" and "root" 
        except KeyError:
            assert msg != "best move"
            self._reset(move)

    def _get_meta_pack(self) -> dict:
        return {
            "explore_weight": self.explore_weight,
            "h_weight": self.h_weight,
            "h_decay": self.h_decay,
        }

    def _reset(self, move: Move):
        self.root_state.play(move)
        self.root = Node(
            **self._get_meta_pack(),
            owner=self.root_state.current_player, 
            plays=self.root_state.get_play_cnt(self.root_state.current_player),
        )
        self.reset_cnt += 1
        