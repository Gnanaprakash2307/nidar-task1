import cv2
from collections import deque

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread("img/maze.png")

if image is None:
    print("ERROR: Could not load img/maze.png")
    exit()

# ============================================================
# 2. GRAYSCALE + THRESHOLD
# ============================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray,
    128,
    255,
    cv2.THRESH_BINARY
)

# ============================================================
# 3. DETECT MAZE BOUNDARY
# ============================================================

black_pixels = binary < 128

ys, xs = black_pixels.nonzero()

maze_x1 = xs.min()
maze_x2 = xs.max()

maze_y1 = ys.min()
maze_y2 = ys.max()

print("Maze boundary:")
print("X:", maze_x1, "to", maze_x2)
print("Y:", maze_y1, "to", maze_y2)

# ============================================================
# 4. MAZE CONFIGURATION
# ============================================================

GRID_ROWS = 20
GRID_COLS = 20

maze_width = maze_x2 - maze_x1 + 1
maze_height = maze_y2 - maze_y1 + 1

cell_width = maze_width / GRID_COLS
cell_height = maze_height / GRID_ROWS

print("Maze size:", maze_width, "x", maze_height)
print("Cell size:", cell_width, "x", cell_height)

# ============================================================
# 5. CELL CENTER
# ============================================================

def cell_center(row, col):

    x = int(
        maze_x1 +
        (col + 0.5) * cell_width
    )

    y = int(
        maze_y1 +
        (row + 0.5) * cell_height
    )

    return x, y


# ============================================================
# 6. CHECK DOORWAY BETWEEN CELLS
# ============================================================

def doorway_is_open(row1, col1, row2, col2):

    x1, y1 = cell_center(row1, col1)
    x2, y2 = cell_center(row2, col2)

    # --------------------------------------------------------
    # Moving horizontally
    # --------------------------------------------------------

    if row1 == row2:

        boundary_x = int(
            maze_x1 +
            max(col1, col2) * cell_width
        )

        center_y = y1

        # Check a small rectangle around the doorway
        x_start = max(0, boundary_x - 4)
        x_end = min(binary.shape[1], boundary_x + 5)

        y_start = max(0, center_y - 10)
        y_end = min(binary.shape[0], center_y + 11)

        region = binary[
            y_start:y_end,
            x_start:x_end
        ]

    # --------------------------------------------------------
    # Moving vertically
    # --------------------------------------------------------

    else:

        boundary_y = int(
            maze_y1 +
            max(row1, row2) * cell_height
        )

        center_x = x1

        x_start = max(0, center_x - 10)
        x_end = min(binary.shape[1], center_x + 11)

        y_start = max(0, boundary_y - 4)
        y_end = min(binary.shape[0], boundary_y + 5)

        region = binary[
            y_start:y_end,
            x_start:x_end
        ]

    # --------------------------------------------------------
    # Calculate white percentage
    # --------------------------------------------------------

    white_ratio = float(
        (region == 255).mean()
    )

    # Debug information
    return white_ratio > 0.45


# ============================================================
# 7. BUILD GRAPH
# ============================================================

graph = {}

for row in range(GRID_ROWS):

    for col in range(GRID_COLS):

        current = (row, col)

        graph[current] = []

        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        if col < GRID_COLS - 1:

            if doorway_is_open(
                row,
                col,
                row,
                col + 1
            ):

                graph[current].append(
                    (row, col + 1)
                )

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        if col > 0:

            if doorway_is_open(
                row,
                col,
                row,
                col - 1
            ):

                graph[current].append(
                    (row, col - 1)
                )

        # ----------------------------------------------------
        # DOWN
        # ----------------------------------------------------

        if row < GRID_ROWS - 1:

            if doorway_is_open(
                row,
                col,
                row + 1,
                col
            ):

                graph[current].append(
                    (row + 1, col)
                )

        # ----------------------------------------------------
        # UP
        # ----------------------------------------------------

        if row > 0:

            if doorway_is_open(
                row,
                col,
                row - 1,
                col
            ):

                graph[current].append(
                    (row - 1, col)
                )


# ============================================================
# 8. START + GOAL
# ============================================================

start = (0, 0)
goal = (19, 19)

print()
print("Start:", start)
print("Goal:", goal)


# ============================================================
# 9. GRAPH DEBUG
# ============================================================

print()
print("=" * 40)
print("GRAPH DEBUG")
print("=" * 40)

print("Start neighbors:", graph[start])
print("Goal neighbors:", graph[goal])

# ============================================================
# 10. BFS
# ============================================================

def bfs(graph, start, goal):

    queue = deque()

    queue.append(start)

    visited = set()

    visited.add(start)

    parent = {}

    parent[start] = None

    while queue:

        current = queue.popleft()

        # Goal reached
        if current == goal:
            break

        for neighbor in graph[current]:

            if neighbor not in visited:

                visited.add(neighbor)

                parent[neighbor] = current

                queue.append(neighbor)

    # --------------------------------------------------------
    # Goal unreachable
    # --------------------------------------------------------

    if goal not in parent:

        return None

    # --------------------------------------------------------
    # Reconstruct path
    # --------------------------------------------------------

    path = []

    current = goal

    while current is not None:

        path.append(current)

        current = parent[current]

    path.reverse()

    return path


# ============================================================
# 11. SOLVE
# ============================================================

path = bfs(
    graph,
    start,
    goal
)

if path is None:

    print()
    print("NO PATH FOUND!")

else:

    print()
    print("=" * 40)
    print("PATH FOUND!")
    print("=" * 40)

    print("Path length:", len(path))

    print()
    print("Path:")

    for cell in path:

        print(cell)


# ============================================================
# 12. DRAW SOLUTION
# ============================================================

result = image.copy()

if path is not None:

    points = []

    for row, col in path:

        x, y = cell_center(row, col)

        points.append(
            (x, y)
        )

    # --------------------------------------------------------
    # Draw path
    # --------------------------------------------------------

    for i in range(len(points) - 1):

        cv2.line(
            result,
            points[i],
            points[i + 1],
            (0, 0, 255),
            5
        )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    sx, sy = cell_center(*start)

    cv2.circle(
        result,
        (sx, sy),
        8,
        (255, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------

    gx, gy = cell_center(*goal)

    cv2.circle(
        result,
        (gx, gy),
        8,
        (0, 255, 0),
        -1
    )


# ============================================================
# 13. RESIZE FOR DISPLAY
# ============================================================

display_width = 600
display_height = 600

small_result = cv2.resize(
    result,
    (display_width, display_height)
)


# ============================================================
# 14. DISPLAY
# ============================================================

cv2.imshow(
    "BFS Maze Solution",
    small_result
)

cv2.waitKey(0)

cv2.destroyAllWindows()