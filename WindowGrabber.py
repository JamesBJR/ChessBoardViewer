import pyautogui
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

def find_chessboard(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2:  # Check if the contour is roughly square
                return (x, y, w, h)
    return None

class ChessBoardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Board Finder")

        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        self.button = tk.Button(root, text="Take Screenshot", command=self.take_screenshot_and_find_chessboard)
        self.button.pack(pady=20)

        self.canvas = tk.Canvas(root, width=800, height=800)
        self.canvas.pack()

        # Placeholder label for the image
        self.image_label = tk.Label(root)
        self.image_label.pack()

    def take_screenshot_and_find_chessboard(self):
        screenshot = take_screenshot()
        chessboard_coords = find_chessboard(screenshot)
        
        if chessboard_coords:
            x, y, w, h = chessboard_coords
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x + w, y + h
            messagebox.showinfo("Result", f"Chessboard found from ({self.start_x}, {self.start_y}) to ({self.end_x}, {self.end_y})")
            
            # Crop the screenshot to the chessboard area
            cropped_image = screenshot[y:y+h, x:x+w]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image = ImageTk.PhotoImage(cropped_image)
            
            # Resize canvas to fit the cropped image
            self.canvas.config(width=w, height=h)
            
            # Display the cropped image in the label
            self.image_label.config(image=cropped_image)
            self.image_label.image = cropped_image  # Keep a reference to avoid garbage collection
        else:
            messagebox.showinfo("Result", "Chessboard not found on the screen.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessBoardApp(root)
    root.mainloop()
