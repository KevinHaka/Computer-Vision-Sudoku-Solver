# Sudoku Screen Solver

Computer-Vision-Sudoku-Solver — screen-capture Sudoku solver using OpenCV + EasyOCR (computer vision pipeline).

A simple script that:
1. Captures a full-screen screenshot.
2. Locates and crops the Sudoku grid.
3. Reads digits with OCR (easyocr).
4. Solves the puzzle via backtracking.
5. Renders (or saves) the solution filling only the originally empty cells.

## Hotkeys
- `Alt + S`: Capture & solve (runs asynchronously; ignores re-trigger while busy)
- `Alt + Q`: Stop the hotkey listener


## Environment & Installation
Create and activate a virtual environment (Windows Command Prompt):
```powershell
python -m venv venv
venv\Scripts\activate
```
Install dependencies:
```powershell
pip install -r requirements.txt
```
## Run
```powershell
python sudokuSolver.py
```
Press `Alt+S` while the Sudoku web page is fully visible. If an OpenCV GUI window cannot be created, the solved image is saved as `sudoku_solution.png`.

## Tested On
- https://www.websudoku.com/
- https://sudoku.com/

## License
This project is licensed under the MIT License - see the LICENSE file for details.