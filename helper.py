import gym
import numpy as np
from gym import spaces
import math
from math import floor
import random
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Board, Direction
from typing import Union, Tuple, Dict, List, Generator
from kaggle_environments.helpers import Point
from config import (
    GAME_CONFIG
)
from itertools import groupby

def getNearestLargestKore(shipyardPos: Point, board: Board) -> Point:
    #print("finding max kore where shipyardPos: ", shipyardPos)
    me = board.current_player
    size = GAME_CONFIG['size']
    
    max_kore = 0
    max_kore_pos = shipyardPos
    # Get largest kore pos
    for i in range(size):
        for j in range(size):
            pos = Point(j, size - 1 - i)
            curr_cell_kore = board.cells[pos].kore
            if curr_cell_kore > max_kore:
                max_kore = curr_cell_kore
                max_kore_pos = pos
    print("max_kore_pos: ", max_kore_pos, " with ", max_kore, " kores")
    return max_kore_pos

def getNearbyLargestKore(shipyardPos: Point, board: Board) -> Point:
    MAX_DIS = 3
    size = GAME_CONFIG['size']
 
    max_kore = 0
    max_kore_pos = shipyardPos
    # Get nearby largest kore pos
    for i in range(size):
        for j in range(size):
            pos = Point(j, size - 1 - i)
            curr_cell_kore = board.cells[pos].kore
            if shipyardPos.distance_to(pos, size) <= MAX_DIS and curr_cell_kore > max_kore:
                max_kore = curr_cell_kore
                max_kore_pos = pos
    print("nearby max_kore_pos: ", max_kore_pos, " with ", max_kore, " kores")
    return max_kore_pos
    
def getFlightPlan(shipyardPos: Point, targetPos: Point, num_ships: int, board: Board) -> str:
    #print("shipyardPos: ", shipyardPos)
    #print("targetPos: ", targetPos)
    me = board.current_player
    
    dx = int(targetPos.x - shipyardPos.x)
    dy = int(targetPos.y - shipyardPos.y)
    print("dx: ", dx, " dy: ", dy)
    
    if(dx > 0 and abs(dx) >= 11):
        dx = -(21-dx)
    if(dy > 0 and abs(dy) >= 11):
        dy = -(21-dy)    
    if(dx < 0 and abs(dx) >= 11):
        dx = 21+dx    
    if(dy < 0 and abs(dy) >= 11):
        dy = 21+dy
    print("tweaked dx: ", dx, " dy: ", dy)
    
    rough_plan= ""
    if dx > 0 and dy > 0:
        rough_plan = rough_plan + "E" * dx
        rough_plan = rough_plan + "N" * dy
    elif dx > 0 and dy < 0:
        rough_plan = rough_plan + "E" * dx
        rough_plan = rough_plan + "S" * -dy
    elif dx < 0 and dy > 0:
        rough_plan = rough_plan + "W" * -dx
        rough_plan = rough_plan + "N" * dy
    elif dx < 0 and dy < 0:
        rough_plan = rough_plan + "W" * -dx
        rough_plan = rough_plan + "S" * -dy
    elif dx == 0:
        if dy > 0:
            rough_plan = rough_plan + "N" * dy
        elif dy < 0:
            rough_plan = rough_plan + "S" * -dy
    elif dy == 0:
        if dx > 0:
            rough_plan = rough_plan + "E" * dx
        elif dx < 0:
            rough_plan = rough_plan + "W" * -dx

    return simplify(rough_plan) + simplify(reversed_str(rough_plan))
    
    counter = 0
    while True:
        counter+=1
        shuffled = ''.join(random.sample(rough_plan,len(rough_plan)))
        
        flight_plan = simplify(shuffled)
        
        # len matches
        if(len(flight_plan) < max_flight_plan_len(num_ships)):
            return flight_plan + simplify(reversed_str(rough_plan))
        # too many times
        elif counter >=10:
            return simplify(rough_plan) + simplify(reversed_str(rough_plan))
        
def max_flight_plan_len(num_ships):
    return floor(2 * np.log(num_ships)) + 1
    
def simplify(plan: str) -> str:
    #print("simplify input: ", plan)
    groups = groupby(plan)
    result = [(label, sum(1 for _ in group)) for label, group in groups]
    simplified = "".join("{}{}".format(label, count-1) for label, count in result)
    simplified = simplified.replace('0', '')
    #print("simplified: ", simplified)
    return simplified

def reversed_str(original: str) -> str:
    #print("original: ", original)
    reversed_str = original[::-1]
    return_str = ""
    for c in reversed_str:
        if(c == "N"):
            return_str = return_str + "S"
        elif(c == "S"):
            return_str = return_str + "N"
        elif(c == "W"):
            return_str = return_str + "E"
        elif(c == "E"):
            return_str = return_str + "W"
    #print("reversed string: ", return_str)
    return return_str   
