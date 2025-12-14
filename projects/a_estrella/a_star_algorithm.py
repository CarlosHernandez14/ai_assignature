import math
import heapq
import pygame
from itertools import count
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

# Window config
WINDOW_SIZE = 800
GRID_DIM = 11  # number of rows and columns (square grid)

# Movement costs (integer scaled: orthogonal=10, diagonal=14)
ORTHO_STEP = 10.0
DIAG_STEP = 14.0

# Colors
COLOR_BG = (245, 245, 245)       # background
COLOR_GRID = (200, 200, 200)     # grid lines
COLOR_EMPTY = (255, 255, 255)    # default tile
COLOR_WALL = (40, 40, 40)        # wall/obstacle
COLOR_START = (0, 158, 115)      # start tile
COLOR_GOAL = (213, 94, 0)        # goal tile
COLOR_OPEN = (0, 114, 178)       # frontier (open set) SKY BLUE
COLOR_CLOSED = (237, 237, 237)   # visited (closed set) LIGHT GRAY
COLOR_PATH = (240, 228, 66)      # final path

# Directions: 8-neighborhood (dx, dy, cost)
NEIGHBOR_OFFSETS: Tuple[Tuple[int, int, float], ...] = (
    (-1,  0, ORTHO_STEP),  # up
    ( 1,  0, ORTHO_STEP),  # down
    ( 0, -1, ORTHO_STEP),  # left
    ( 0,  1, ORTHO_STEP),  # right
    (-1, -1, DIAG_STEP),   # up-left
    (-1,  1, DIAG_STEP),   # up-right
    ( 1, -1, DIAG_STEP),   # down-left
    ( 1,  1, DIAG_STEP),   # down-right
)


class Tile:
    """
    Represents a single cell in the grid with a visual state.
    """
    __slots__ = ("row", "col", "size", "rect", "_color", "is_barrier")

    def __init__(self, row: int, col: int, size: int) -> None:
        self.row = row
        self.col = col
        self.size = size
        # Normalized mapping: x = col * size, y = row * size
        x = col * size
        y = row * size
        self.rect = pygame.Rect(x, y, size, size)
        self._color = COLOR_EMPTY
        self.is_barrier = False

    def reset(self) -> None:
        self._color = COLOR_EMPTY
        self.is_barrier = False

    def set_wall(self) -> None:
        self._color = COLOR_WALL
        self.is_barrier = True

    def set_start(self) -> None:
        self._color = COLOR_START
        self.is_barrier = False

    def set_goal(self) -> None:
        self._color = COLOR_GOAL
        self.is_barrier = False

    def set_open(self) -> None:
        self._color = COLOR_OPEN

    def set_closed(self) -> None:
        self._color = COLOR_CLOSED

    def set_path(self) -> None:
        self._color = COLOR_PATH

    def is_start(self) -> bool:
        return self._color == COLOR_START

    def is_goal(self) -> bool:
        return self._color == COLOR_GOAL

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self._color, self.rect)


class Board:
    """
    Encapsulates the grid (2D tiles) and all spatial/visual utilities.
    """
    def __init__(self, rows: int, cols: int, window_size: int) -> None:
        assert rows == cols, "Board currently requires a square grid"
        self.rows = rows
        self.cols = cols
        self.window_size = window_size
        self.cell_size = window_size // rows
        self.tiles: List[List[Tile]] = [
            [Tile(r, c, self.cell_size) for c in range(cols)] for r in range(rows)
        ]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and not self.tiles[r][c].is_barrier

    def iter_neighbors(self, r: int, c: int) -> Iterable[Tuple[int, int, float]]:
        """
        Yields (nr, nc, move_cost) for 8-directional neighbors that are walkable.
        Corner-cutting is allowed as long as the diagonal target is not a wall.
        """
        for dr, dc, cost in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if self.is_walkable(nr, nc):
                yield nr, nc, cost

    def coords_from_mouse(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Convert pixel coordinates to (row, col) indices.
        """
        x, y = pos
        col = x // self.cell_size
        row = y // self.cell_size
        if self.in_bounds(row, col):
            return row, col
        return None

    def clear(self) -> None:
        for row in self.tiles:
            for t in row:
                t.reset()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(COLOR_BG)
        for row in self.tiles:
            for t in row:
                t.draw(surface)
        self._draw_grid(surface)
        pygame.display.update()

    def _draw_grid(self, surface: pygame.Surface) -> None:
        # Draw grid lines
        for r in range(self.rows + 1):
            y = r * self.cell_size
            pygame.draw.line(surface, COLOR_GRID, (0, y), (self.window_size, y))
        for c in range(self.cols + 1):
            x = c * self.cell_size
            pygame.draw.line(surface, COLOR_GRID, (x, 0), (x, self.window_size))


def octile_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Heuristic suitable for 8-directional grids.
    """
    r1, c1 = a
    r2, c2 = b
    dx = abs(r1 - r2)
    dy = abs(c1 - c2)
    return DIAG_STEP * min(dx, dy) + ORTHO_STEP * abs(dx - dy)


def reconstruct_path(
    parents: Dict[Tuple[int, int], Tuple[int, int]],
    current: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Recreate path from goal to start using the 'parents' map.
    Returns the path from start to goal inclusive.
    """
    out: List[Tuple[int, int]] = [current]
    while current in parents:
        current = parents[current]
        out.append(current)
    out.reverse()
    return out


def solve_a_star(
    board: Board,
    start: Tile,
    goal: Tile,
    draw: Callable[[], None],
) -> bool:
    """
    Runs A* on the provided board while visualizing progress.
    Returns True if a path is found, False otherwise.
    """
    start_rc = (start.row, start.col)
    goal_rc = (goal.row, goal.col)

    # Algorithm state
    g_score: Dict[Tuple[int, int], float] = {
        (r, c): float("inf")
        for r in range(board.rows)
        for c in range(board.cols)
    }
    f_score: Dict[Tuple[int, int], float] = {
        (r, c): float("inf")
        for r in range(board.rows)
        for c in range(board.cols)
    }
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}

    g_score[start_rc] = 0.0
    f_score[start_rc] = octile_distance(start_rc, goal_rc)

    tie = count()
    # heap entries: (f, h, -g, tie_idx, (r, c))
    open_heap: List[Tuple[float, float, float, int, Tuple[int, int]]] = []
    h0 = octile_distance(start_rc, goal_rc)
    heapq.heappush(open_heap, (f_score[start_rc], h0, -g_score[start_rc], next(tie), start_rc))
    in_open: Set[Tuple[int, int]] = {start_rc}
    in_closed: Set[Tuple[int, int]] = set()

    # Ensure start/goal colors are correct
    start.set_start()
    goal.set_goal()

    while open_heap:
        # Allow quitting during algorithm
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        f_popped, h_popped, neg_g_popped, _, current = heapq.heappop(open_heap)
        if current in in_open:
            in_open.remove(current)
        # Skip stale entries whose priority no longer matches the best-known g
        if -neg_g_popped != g_score[current]:
            continue

        if current in in_closed:
            # Skip entries made obsolete by a better path
            continue

        current_g = g_score[current]

        if current == goal_rc:
            # Build and paint final path (excluding endpoints)
            path = reconstruct_path(parents, current)
            for (r, c) in path:
                tile = board.tiles[r][c]
                if tile is not start and tile is not goal:
                    tile.set_path()
            start.set_start()
            goal.set_goal()
            draw()
            return True

        in_closed.add(current)
        cr, cc = current

        # Visual feedback: mark current as closed if not start or goal
        cur_tile = board.tiles[cr][cc]
        if (cur_tile is not start) and (cur_tile is not goal):
            cur_tile.set_closed()

        # Explore neighbors
        for nr, nc, step_cost in board.iter_neighbors(cr, cc):
            neighbor = (nr, nc)
            if neighbor in in_closed:
                continue

            tentative = current_g + step_cost
            if tentative < g_score[neighbor]:
                parents[neighbor] = current
                g_score[neighbor] = tentative
                h = octile_distance(neighbor, goal_rc)
                f = tentative + h
                f_score[neighbor] = f

                # Always reinsert on improvement (simulate decrease-key)
                heapq.heappush(open_heap, (f, h, -g_score[neighbor], next(tie), neighbor))
                in_open.add(neighbor)
                n_tile = board.tiles[nr][nc]
                if (n_tile is not start) and (n_tile is not goal):
                    n_tile.set_open()

        draw()

    return False  # no path found


def run() -> None:
    pygame.init()
    pygame.display.set_caption("A* Pathfinding Visualizer")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()

    board = Board(GRID_DIM, GRID_DIM, WINDOW_SIZE)

    start: Optional[Tile] = None
    goal: Optional[Tile] = None

    running = True
    while running:
        clock.tick(60)
        board.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse interactions
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                # Left click: start -> goal -> walls
                pos = pygame.mouse.get_pos()
                rc = board.coords_from_mouse(pos)
                if rc is not None:
                    r, c = rc
                    tile = board.tiles[r][c]
                    if start is None and tile is not goal:
                        start = tile
                        start.set_start()
                    elif goal is None and tile is not start:
                        goal = tile
                        goal.set_goal()
                    elif tile is not start and tile is not goal:
                        tile.set_wall()

            if pygame.mouse.get_pressed(num_buttons=3)[2]:
                # Right click: reset tile; update start/goal references
                pos = pygame.mouse.get_pos()
                rc = board.coords_from_mouse(pos)
                if rc is not None:
                    r, c = rc
                    tile = board.tiles[r][c]
                    was_start = tile is start
                    was_goal = tile is goal
                    tile.reset()
                    if was_start:
                        start = None
                    if was_goal:
                        goal = None

            # Keyboard interactions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and goal:
                    # Run A* visualization
                    # Reaffirm colors before solve
                    start.set_start()
                    goal.set_goal()
                    solve_a_star(board, start, goal, lambda: board.draw(screen))

                if event.key == pygame.K_c:
                    # Clear everything
                    board.clear()
                    start = None
                    goal = None

        # board.draw(screen) handled at top of loop

    pygame.quit()


if __name__ == "__main__":
    run()
