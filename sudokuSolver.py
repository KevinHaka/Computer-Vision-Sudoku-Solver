# Needed Libraries
import cv2 as cv 
import numpy as np
import easyocr

from PIL import ImageGrab
from pynput import keyboard
from threading import Lock, Thread

reader = easyocr.Reader(["en"], gpu=False)

# Lock to prevent re-entrant activation while solving
_solve_lock = Lock()

# Determine the size of the Sudoku grid
grid_size = 9

# Function to print the Sudoku grid in a formatted way
def printSudoku(sudoku):
    length = len(sudoku)

    for row in range(length):
        print("\n|","-" * (4*length-1), "|", sep="")
        print("|", end=" ")
        
        for col in range(length):
            if sudoku[row, col]:
                print(sudoku[row, col], end=" | ")
            else: print(" ", end=" | ")

    print("\n|","-" * (4*length-1), "|", sep="")

# Image Processing
def imageProcessing(image):
    # Convert the image to grayscale
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply GaussianBlur and adaptive thresholding to the image
    cv.GaussianBlur(grayScale, (3, 3), 0, grayScale)
    cv.adaptiveThreshold(grayScale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2, grayScale)

    # Find the contours
    contours, _ = cv.findContours(grayScale, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Short all the contours by area
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

    # Contour of interest
    contour_of_interest = sorted_contours[1]

    # Create a mask for the contour of interest
    mask = np.zeros_like(grayScale)
    cv.drawContours(mask, [contour_of_interest], -1, 255, thickness=cv.FILLED)

    # Find the bounding rectangle of the contour
    x, y, w, h = cv.boundingRect(contour_of_interest)

    # Crop the image to the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

# Function to extract the Sudoku grid from the image
def extract_sudoku_from_image(cropped_image, reader, num_cells):
    # Calculate the step size for each cell
    h, w = cropped_image.shape[:2]
    x_step = w / num_cells
    y_step = h / num_cells

    # Border filter
    x_corrector = x_step * 0.2
    y_corrector = y_step * 0.2

    # Initialize the sudoku grid
    sudoku = np.zeros((num_cells, num_cells), dtype=int)
    empty_cells = []

    for row in range(num_cells):
        for col in range(num_cells):
            x1 = int(round(col * x_step) + x_corrector) 
            x2 = int(round(x1 + x_step) - x_corrector)

            y1 = int(round(row * y_step) + y_corrector)
            y2 = int(round(y1 + y_step) - y_corrector)

            cell = cropped_image[y1:y2, x1:x2]

            # Preprocess the cell image
            cell = cv.bilateralFilter(cell, 9, 75, 75)
            gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (3, 3), 0)
            binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            resized = cv.resize(binary, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC)

            # Read text from the cell with adjusted parameters
            result = reader.readtext(
                resized,
                allowlist="123456789",
                contrast_ths=0.,
                text_threshold=0.1,
                low_text=0.05,
            )
        
            # If the text is found, add it to the sudoku grid
            if result and result[0][1]:
                sudoku[row, col] = int(result[0][1])
            else: empty_cells.append((row, col))
            
    return sudoku, empty_cells

def find_empty_location(grid):
    length = len(grid)
    for row in range(length):
        for col in range(length):
            if grid[row][col] == 0:
                return row, col
    return None
            
def is_valid(grid, row, col, num):
    length = len(grid)
    length_sqrt = int(length ** 0.5)

    # Check if the number is already in the row or column
    if (num in grid[row]) or (num in grid[:, col]):
        return False

    # Check if the number is already in the box 
    start_row = row - row % length_sqrt
    start_col = col - col % length_sqrt

    if num in grid[start_row:start_row + length_sqrt, start_col:start_col + length_sqrt]:
        return False
    return True

def sudokuSolver(sudoku):
    sudoku = np.array(sudoku)
    row_col = find_empty_location(sudoku)

    if row_col: row, col = row_col
    else: return sudoku

    length = len(sudoku)
    for n in range(1, length+1):
        if is_valid(sudoku, row, col, n):
            sudoku[row][col] = n

            solution = sudokuSolver(sudoku)
            if not type(solution) == bool: return solution

            sudoku[row][col] = 0
    return False

def sudoku_to_image(cropped_image, solution, empty_cells):
    sudoku_image = cropped_image.copy()
    width, height = cv.getTextSize("0", cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

    length = len(solution)
    h, w = cropped_image.shape[:2]
    x_step = w / length
    y_step = h / length
    center_x = (x_step - width) / 2
    center_y = (y_step + height) / 2

    for row, col in empty_cells:
        x = int(col * x_step + center_x)
        y = int(row * y_step + center_y)

        cv.putText(
            img = sudoku_image,
            text = str(solution[row, col]),
            org = (x, y),
            fontFace = cv.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (0, 0, 255),
            thickness = 2,
            lineType = cv.LINE_AA,
            bottomLeftOrigin = False 
        )

    return sudoku_image

def _solve_async():
    try:
        print("- Screen captured...")
        image = np.array(ImageGrab.grab())

        print("- Image processing...")
        cropped_image = imageProcessing(image)

        print("- Extracting Sudoku...")
        sudoku, empty_cells = extract_sudoku_from_image(cropped_image, reader, grid_size)

        print("- Solving Sudoku...")
        solution = sudokuSolver(sudoku)

        if isinstance(solution, bool):
            printSudoku(sudoku)
            print("- No solution found...")
            return

        print("\nDisplaying the initial Sudoku...")
        printSudoku(sudoku)
        print("\nDisplaying the solved Sudoku...")
        printSudoku(solution)

        sudoku_image = sudoku_to_image(cropped_image, solution, empty_cells)
        try:
            cv.namedWindow('Solved Sudoku', cv.WINDOW_NORMAL)
            cv.setWindowProperty('Solved Sudoku', cv.WND_PROP_TOPMOST, 1)
            cv.imshow('Solved Sudoku', sudoku_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        except cv.error:
            cv.imwrite("sudoku_solution.png", sudoku_image)
            print("Saved sudoku_solution.png (no GUI backend).")
        print("Done!\n")
    finally:
        if _solve_lock.locked():
            _solve_lock.release()

def on_activate():
    if not _solve_lock.acquire(blocking=False):
        print("Already solving... please wait.")
        return
    Thread(target=_solve_async, daemon=True).start()

def on_stop():
    print("Stopping listener...")
    listener.stop()

if __name__ == "__main__":
    print("Press <alt>+s to start solving Sudoku or <alt>+q to stop the listener.")
    with keyboard.GlobalHotKeys({
        '<alt>+s': on_activate,
        '<alt>+q': on_stop
    }) as listener: listener.join()