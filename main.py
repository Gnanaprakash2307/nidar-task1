import cv2

# Load maze
image = cv2.imread("img/maze.png")

if image is None:
    print("Error: Could not load image.")
    exit()

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Invert binary image
# Black objects become white
black_objects = cv2.bitwise_not(binary)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    black_objects,
    connectivity=8
)

print("Number of components:", num_labels)

# Store possible small components
candidates = []

for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    # Ignore very large components
    if area < 1500 and w < 60 and h < 60:
        candidates.append({
            "label": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area
        })

print("\nSmall component candidates:")

for c in candidates:
    print(c)


# -----------------------------------
# Find S: closest candidate to top-left
# -----------------------------------

if candidates:

    start = min(
        candidates,
        key=lambda c: c["x"] + c["y"]
    )

    # -----------------------------------
    # Find G: closest candidate to bottom-right
    # -----------------------------------

    image_height, image_width = binary.shape

    goal = min(
        candidates,
        key=lambda c:
        (image_width - c["x"]) +
        (image_height - c["y"])
    )

    print("\nDetected Start:")
    print(start)

    print("\nDetected Goal:")
    print(goal)

    # Draw results
    result = image.copy()

    # Start rectangle
    cv2.rectangle(
        result,
        (start["x"], start["y"]),
        (
            start["x"] + start["w"],
            start["y"] + start["h"]
        ),
        (0, 0, 255),
        3
    )

    # Goal rectangle
    cv2.rectangle(
        result,
        (goal["x"], goal["y"]),
        (
            goal["x"] + goal["w"],
            goal["y"] + goal["h"]
        ),
        (255, 0, 0),
        3
    )

    # Display
    display_size = (600, 600)
    small_result = cv2.resize(result, display_size)

    cv2.imshow(
        "Detected Start and Goal",
        small_result
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No suitable candidates found.")
