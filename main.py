import cv2

# ==========================================
# 1. Load image
# ==========================================

image = cv2.imread("img/maze.png")

if image is None:
    print("Error: Could not load image.")
    exit()

# ==========================================
# 2. Convert to grayscale
# ==========================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==========================================
# 3. Threshold
# ==========================================

_, binary = cv2.threshold(
    gray,
    128,
    255,
    cv2.THRESH_BINARY
)

# ==========================================
# 4. Maze configuration
# ==========================================

GRID_ROWS = 20
GRID_COLS = 20

height, width = binary.shape

cell_width = width / GRID_COLS
cell_height = height / GRID_ROWS

print("Image size:", width, "x", height)
print("Cell size:", cell_width, "x", cell_height)


# ==========================================
# 5. Find center of a cell
# ==========================================

def cell_center(row, col):

    x = int((col + 0.5) * cell_width)
    y = int((row + 0.5) * cell_height)

    return x, y


# ==========================================
# 6. Check if two cells are connected
# ==========================================

def can_move(row1, col1, row2, col2):

    x1, y1 = cell_center(row1, col1)
    x2, y2 = cell_center(row2, col2)

    # Number of points to check between
    # the two cell centers
    num_samples = 20

    for i in range(1, num_samples):

        t = i / num_samples

        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))

        # Black pixel = wall
        if binary[y, x] < 128:
            return False

    return True


# ==========================================
# 7. Test maze connections
# ==========================================

print("\nTesting cell connections:\n")

for row in range(GRID_ROWS):

    for col in range(GRID_COLS):

        # ------------------------------
        # Check RIGHT
        # ------------------------------

        if col < GRID_COLS - 1:

            if can_move(row, col, row, col + 1):

                print(
                    f"({row},{col}) -> RIGHT -> "
                    f"({row},{col + 1})"
                )

        # ------------------------------
        # Check DOWN
        # ------------------------------

        if row < GRID_ROWS - 1:

            if can_move(row, col, row + 1, col):

                print(
                    f"({row},{col}) -> DOWN -> "
                    f"({row + 1},{col})"
                )


# ==========================================
# 8. Visualize the grid
# ==========================================

grid_image = image.copy()

# Draw horizontal grid lines

for row in range(GRID_ROWS + 1):

    y = int(row * cell_height)

    cv2.line(
        grid_image,
        (0, y),
        (width, y),
        (0, 0, 255),
        1
    )


# Draw vertical grid lines

for col in range(GRID_COLS + 1):

    x = int(col * cell_width)

    cv2.line(
        grid_image,
        (x, 0),
        (x, height),
        (0, 0, 255),
        1
    )


# ==========================================
# 9. Draw cell centers
# ==========================================

for row in range(GRID_ROWS):

    for col in range(GRID_COLS):

        x, y = cell_center(row, col)

        cv2.circle(
            grid_image,
            (x, y),
            2,
            (255, 0, 0),
            -1
        )


# ==========================================
# 10. Resize for display
# ==========================================

display_size = (600, 600)

small_grid = cv2.resize(
    grid_image,
    display_size
)


# ==========================================
# 11. Display
# ==========================================

cv2.imshow(
    "20x20 Maze Grid",
    small_grid
)

cv2.waitKey(0)
cv2.destroyAllWindows()