import cv2

# Load maze
image = cv2.imread("img/maze.png")

if image is None:
    print("Error: Could not load image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale to binary
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Resize only for display
display_size = (600, 600)

small_original = cv2.resize(image, display_size)
small_binary = cv2.resize(binary, display_size)

# Display
cv2.imshow("Original Maze", small_original)
cv2.imshow("Binary Maze", small_binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
