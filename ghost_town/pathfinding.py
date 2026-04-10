"""BFS pathfinding for Ghost Town agents using the Tiled map's Wall layer."""
from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

Coordinate = Tuple[int, int]

# Wall tiles extracted from ghost_town.json Wall layer (140×100 map).
# Agents cannot enter these tiles.
_WALL_TILES: FrozenSet[Coordinate] = frozenset({
    (69, 56), (92, 58), (47, 53), (44, 91), (27, 59), (42, 57), (82, 93),
    (85, 55), (88, 54), (37, 70), (105, 70), (102, 71), (25, 59), (42, 75),
    (46, 57), (82, 56), (73, 53), (38, 53), (37, 88), (48, 8), (31, 56),
    (22, 53), (83, 94), (43, 85), (40, 13), (69, 58), (47, 55), (73, 74),
    (90, 90), (19, 57), (33, 56), (85, 57), (27, 70), (102, 73), (45, 85),
    (30, 75), (41, 8), (39, 53), (15, 53), (37, 90), (92, 90), (46, 13),
    (38, 9), (49, 9), (21, 57), (23, 54), (86, 58), (89, 54), (29, 70),
    (40, 70), (106, 70), (81, 71), (103, 74), (47, 57), (17, 53), (93, 54),
    (27, 72), (40, 91), (83, 71), (108, 70), (15, 55), (38, 57), (72, 52),
    (33, 70), (38, 11), (49, 11), (23, 56), (38, 75), (42, 91), (69, 53),
    (45, 53), (70, 52), (80, 53), (71, 72), (27, 56), (90, 94), (93, 56),
    (82, 90), (39, 85), (46, 91), (47, 13), (30, 70), (83, 73), (108, 72),
    (80, 74), (37, 85), (44, 8), (29, 56), (15, 57), (81, 57), (92, 94),
    (41, 85), (43, 91), (71, 74), (73, 71), (82, 74), (17, 57), (93, 58),
    (36, 70), (85, 54), (102, 70), (75, 53), (77, 71), (82, 92), (28, 75),
    (43, 57), (37, 87), (75, 71), (42, 13), (26, 59), (32, 75), (73, 55),
    (87, 54), (104, 70), (79, 71), (72, 74), (45, 57), (47, 54), (37, 53),
    (91, 54), (46, 86), (40, 75), (30, 56), (82, 94), (85, 56), (76, 53),
    (75, 55), (102, 72), (38, 91), (32, 59), (86, 94), (33, 58), (42, 70),
    (18, 57), (73, 57), (93, 90), (21, 53), (44, 85), (40, 8), (47, 56),
    (22, 57), (37, 55), (77, 57), (88, 94), (39, 91), (85, 58), (102, 74),
    (75, 57), (80, 71), (84, 90), (42, 8), (15, 54), (25, 56), (16, 53),
    (79, 57), (82, 53), (32, 70), (42, 72), (27, 74), (46, 8), (39, 57),
    (87, 58), (93, 92), (71, 71), (82, 71), (38, 13), (49, 13), (39, 75),
    (37, 57), (93, 55), (69, 55), (91, 58), (34, 70), (46, 90), (37, 75),
    (41, 57), (83, 72), (44, 53), (43, 8), (15, 56), (26, 56), (25, 58),
    (42, 74), (82, 55), (73, 52), (108, 74), (41, 75), (45, 91), (89, 94),
    (93, 94), (81, 74), (71, 73), (85, 90), (38, 70), (39, 13), (28, 56),
    (69, 57), (80, 57), (82, 91), (40, 85), (47, 8), (32, 56), (84, 94),
    (41, 13), (16, 57), (82, 57), (73, 54), (35, 70), (76, 71), (74, 74),
    (42, 85), (37, 89), (91, 90), (49, 8), (38, 8), (20, 57), (23, 53),
    (39, 70), (46, 85), (31, 75), (75, 54), (78, 71), (33, 57), (44, 57),
    (41, 70), (90, 54), (28, 70), (107, 70), (27, 71), (73, 56), (37, 91),
    (40, 53), (39, 8), (38, 10), (49, 10), (31, 59), (23, 55), (37, 54),
    (41, 91), (69, 52), (92, 54), (85, 94), (75, 56), (33, 59), (27, 73),
    (76, 57), (87, 94), (81, 53), (46, 53), (71, 52), (42, 71), (108, 71),
    (73, 58), (93, 91), (91, 94), (83, 90), (48, 13), (38, 12), (49, 12),
    (23, 57), (37, 56), (78, 57), (69, 54), (31, 70), (72, 71), (38, 85),
    (45, 8), (93, 57), (27, 75), (25, 57), (42, 73), (83, 74), (82, 54),
    (74, 71), (108, 73), (107, 74), (37, 86), (93, 93), (40, 57), (86, 54),
    (103, 70), (29, 75),
})

_MAP_WIDTH = 140
_MAP_HEIGHT = 100


def find_path(start: Coordinate, end: Coordinate) -> List[Coordinate]:
    """BFS shortest path from start to end avoiding wall tiles.

    Returns a list of tiles from start (exclusive) to end (inclusive).
    If no path exists, returns [end] as a fallback (original behaviour).
    """
    if start == end:
        return []

    # If end is a wall tile, find nearest walkable neighbour
    if end in _WALL_TILES:
        end = _nearest_walkable(end)
        if end is None or end == start:
            return []

    queue: deque[List[Coordinate]] = deque([[start]])
    visited: set[Coordinate] = {start}

    while queue:
        path = queue.popleft()
        cx, cy = path[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            npos = (nx, ny)
            if not (0 <= nx < _MAP_WIDTH and 0 <= ny < _MAP_HEIGHT):
                continue
            if npos in visited or npos in _WALL_TILES:
                continue
            new_path = path + [npos]
            if npos == end:
                return new_path[1:]  # exclude start, include end
            visited.add(npos)
            queue.append(new_path)

    # Fallback: direct step (original behaviour)
    return [end]


def _nearest_walkable(pos: Coordinate) -> Optional[Coordinate]:
    """BFS outward from pos to find the closest non-wall tile."""
    visited: set[Coordinate] = {pos}
    queue: deque[Coordinate] = deque([pos])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            npos = (nx, ny)
            if not (0 <= nx < _MAP_WIDTH and 0 <= ny < _MAP_HEIGHT):
                continue
            if npos in visited:
                continue
            if npos not in _WALL_TILES:
                return npos
            visited.add(npos)
            queue.append(npos)
    return None
