from meta import GameMeta
from utils import get_direction_pattern, out_of_bound
from copy import deepcopy

class Group:
    def __init__(self, cells: set) -> None:
        self.cells = cells

    def get_connect_regions(self):
        connected_regions = []
        nodes = list(deepcopy(self.cells))

        while len(nodes) != 0:
            connected_regions.append(self._dfs(nodes))

        return connected_regions

    def _dfs(self, nodes: list):
        stack = []
        stack.append(nodes.pop())
        connected_region = set()
        
        while len(stack) != 0:
            n = stack.pop()
            connected_region.add(n)

            for d in GameMeta.DIRECTIONS:
                pat_x, pat_y = get_direction_pattern(n, d)
                neighbor = (n[0]+pat_x, n[1]+pat_y)

                if not out_of_bound(neighbor) and neighbor in nodes:
                    stack.append(neighbor)
                    nodes.remove(neighbor)

        return connected_region