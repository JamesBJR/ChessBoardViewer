import os
import sys
import time
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from PIL import ImageGrab, Image, ImageTk
import pyautogui
import cv2
import numpy as np
import joblib
from skimage.feature import hog #Get a fat hog
from stockfish import Stockfish
import chess
import chess.engine
import keyboard
import mouse



class ChessBoardDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Board")
        
        # Set the window icon (ensure the filename and path are correct)
        root.iconbitmap("AppIcon.ico")
        


        # Load previous selection coordinates if available
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.load_coordinates()
        

                        # Create a canvas for the chessboard
        self.canvas = tk.Canvas(self.root, width=(self.end_x - self.start_x), height=(self.end_y - self.start_y))
        self.canvas.place(x=self.start_x, y=self.start_y)
       # Frame for Chessboard Display
        # Rearranged GUI elements for a more organized layout

        # Create a label to display the chessboard image
        self.image_label = tk.Label(root)
        self.image_label.pack(fill="both", expand=True, pady=10, side="right")

        # Create buttons: 'Select Area', 'Analyze Board', 'Restart Stockfish', 'Clear Overlays',
        button_frame1 = tk.Frame(root)
        button_frame1.pack(side="top", pady=10, fill="x")

        self.screenshot_button = tk.Button(button_frame1, text="Select Area", command=self.select_area)
        self.screenshot_button.pack(side="left", padx=5)

        self.analyze_button = tk.Button(button_frame1, text="Analyze Board", command=self.analyze_board)
        self.analyze_button.pack(side="left", padx=5)

        self.restart_button = tk.Button(button_frame1, text="Restart Stockfish", command=self.Restart_stockfish)
        self.restart_button.pack(side="left", padx=5)

        self.clear_overlays_button = tk.Button(button_frame1, text="Clear Overlays", command=self.clear_overlay_boxes)
        self.clear_overlays_button.pack(side="left", padx=5)

        # Create buttons:  'Auto Detect Board',  'Test Mate Moves'
        button_frame2 = tk.Frame(root)
        button_frame2.pack(side="top", pady=10, fill="x")

        self.auto_detect_button = tk.Button(button_frame2, text="Auto Detect Board", command=self.auto_detect_board)
        self.auto_detect_button.pack(side="left", padx=5)

        self.test_highlight_button = tk.Button(button_frame2, text="Test Mate Moves", command=self.test_highlight_mate_moves)
        self.test_highlight_button.pack(side="left", padx=5)

            # Create a frame for the checkboxes
        checkbox1_frame = tk.Frame(root)
        checkbox1_frame.pack(side="top", pady=10, fill="x")

        # Create the checkbox to toggle 'always on top'
        self.keep_on_top_var = tk.BooleanVar(value=False)
        self.keep_on_top_checkbox = tk.Checkbutton(checkbox1_frame, text="Keep on Top", variable=self.keep_on_top_var, command=self.toggle_on_top)
        self.keep_on_top_checkbox.pack(side="left", padx=5)

        # Create a checkbox to select the player's color
        self.player_color_var = tk.StringVar(value="White")
        self.player_color_checkbox = tk.Checkbutton(checkbox1_frame, text="Player is Black", variable=self.player_color_var, onvalue="Black", offvalue="White", command=self.analyze_board)
        self.player_color_checkbox.pack(side="left", padx=5)

        self.eval_box_var = tk.BooleanVar(value=False)
        self.eval_box_checkbox = tk.Checkbutton(checkbox1_frame, text="Evaluation", variable=self.eval_box_var)
        self.eval_box_checkbox.pack(side="left", padx=5)

       # Create a frame for the checkboxes 2
        checkbox2_frame = tk.Frame(root)
        checkbox2_frame.pack(side="top", pady=10, fill="x")

        self.debug_checkbox_var = tk.BooleanVar(value=False)
        self.debug_checkbox = tk.Checkbutton(checkbox2_frame, text="Debug Mode", variable=self.debug_checkbox_var)
        self.debug_checkbox.pack(side="left", padx=5)
        self.debug_checkbox_var.trace("w", self.on_debug_checkbox_change)

        self.reanalyze_var = tk.BooleanVar(value=False)
        self.reanalyze_checkbox = tk.Checkbutton(checkbox2_frame, text="Recapture", variable=self.reanalyze_var)
        self.reanalyze_checkbox.pack(side="left", padx=5)

        self.scroll_hotkey_var = tk.BooleanVar(value=False)
        self.scroll_hotkey_checkbox = tk.Checkbutton(checkbox2_frame, text="Scroll Hotkey", variable=self.scroll_hotkey_var)
        self.scroll_hotkey_checkbox.pack(side="left", padx=5)

        # Create a frame for the checkboxes 3
        checkbox3_frame = tk.Frame(root)
        checkbox3_frame.pack(side="top", pady=10, fill="x")

        self.show_mate_var = tk.BooleanVar(value=False)
        self.show_mate_checkbox = tk.Checkbutton(checkbox2_frame, text="Show Mate", variable=self.show_mate_var)
        self.show_mate_checkbox.pack(side="left", padx=5)

        # Create a frame for evaluation bar
        eval_frame = tk.Frame(root)
        eval_frame.pack(side="top", pady=10, fill="x")

        # Add evaluation bar to show who is winning
        self.evaluation_bar = ttk.Progressbar(eval_frame, orient='horizontal', length=300, mode='determinate')
        self.evaluation_bar.pack(side="top", pady=5)

        # Add label to show numeric evaluation value
        self.eval_label = tk.Label(eval_frame, text="Evaluation: Equal", font=("Arial", 10))
        self.eval_label.pack(side="top", pady=5)

        # Threshold slider for color detection
        slider_frame = tk.Frame(root)
        slider_frame.pack(side="bottom", pady=10, fill="x")

        self.threshold_value = tk.IntVar(value=147)  # Initial value for threshold
        self.threshold_slider = tk.Scale(slider_frame, from_=135, to=165, orient="horizontal", label="       White <---- Color Threshold ----> Black       ",
                                        variable=self.threshold_value, resolution=1, length=300)
        self.threshold_slider.pack(side="top", pady=5)

        # Stockfish Think Time Slider
        self.think_time_slider = tk.Scale(slider_frame, from_=100, to=10000, orient="horizontal", label="Think Time (ms)",
                                        length=300, resolution=1)
        self.think_time_slider.set(500)
        self.think_time_slider.pack(side="top", pady=5)

        # Stockfish Skill Level Slider
        self.fish_skill_slider = tk.Scale(slider_frame, from_=1, to=20, orient="horizontal", label="Skill Level",
                                        length=300, resolution=1)
        self.fish_skill_slider.set(10)
        self.fish_skill_slider.pack(side="top", pady=5)

        # Capture Interval Slider
        self.capture_interval_slider = tk.Scale(slider_frame, from_=100, to=3000, orient="horizontal", label="Capture Interval (ms)",
                                                length=300, resolution=100)
        self.capture_interval_slider.set(5000)
        self.capture_interval_slider.pack(side="top", pady=5)

        # Frame for Console Output
        self.console_frame = tk.Frame(root)
        self.console_output = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, height=10, width=40)
        self.console_output.pack(fill="both", expand=True)
        self.console_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Duplicate print statements to the console output
        self.original_stdout = sys.stdout
        sys.stdout = self

        # Load pre-trained SVM model for piece recognition
        self.svm_model = self.load_svm_model()


        # Initialize Stockfish engine
        stockfish_path = r"C:\GitHubRepos\ChessBoardViewer\stockfish\stockfish-windows-x86-64-avx2.exe"
        self.stockfish_path = stockfish_path
        self.initialize_stockfish()

        # Store the latest screenshot with grid for drawing moves
        self.screenshot_with_grid = None

        # Flag to track if hotkeys are temporarily denied  
        self.hotkey_denied = False  

        # Bind key event to reanalyze board
        keyboard.add_hotkey('a', lambda: self.analyze_board_if_ready())
        keyboard.add_hotkey('s', lambda: self.toggle_player_color())

            # Add hotkey for mouse wheel scroll up to analyze the board
        mouse.hook(self.on_mouse_event)
        pass

        # Start monitoring board changes if recapture is enabled
        self.last_capture = None
        self.monitor_board_changes()

#-------GUI Methods
    def write(self, message): #Write the message to the console output
        # Insert message into console output
        self.console_output.insert(tk.END, message)
        self.console_output.see(tk.END)
        self.original_stdout.write(message)

        # Remove previous highlight
        self.console_output.tag_remove("highlight", "1.0", tk.END)
        
        # Highlight the last message
        last_index = self.console_output.index(tk.END + "-2c linestart")
        self.console_output.tag_add("highlight", last_index, tk.END + "-1c")
        self.console_output.tag_config("highlight", background="yellow", foreground="black")

    def flush(self): #Flush the console output
        self.original_stdout.flush()

    def on_mouse_event(self, event): #Method to handle mouse events
        if self.scroll_hotkey_var.get():
            # Check if the event is a wheel event and if it is a scroll up
            if isinstance(event, mouse.WheelEvent) and event.delta > 0:
                self.analyze_board_if_ready()
            if isinstance(event, mouse.WheelEvent) and event.delta < 0:
                self.analyze_board_if_ready()
            
    def deny_hotkeys_for(self, duration):#Method to deny hotkeys for a set amount of time
        # Method to deny hotkeys for a set amount of time
        def deny():
            self.hotkey_denied = True
            time.sleep(duration)
            self.hotkey_denied = False
        threading.Thread(target=deny).start()

    def analyze_board_if_ready(self): #Method to analyze the board if hotkeys are not denied
        # Only proceed if hotkeys are not denied
        if not self.hotkey_denied:
            self.analyze_board()

    def clear_overlay_boxes(self): #Method to clear overlay boxes from the screen
        # Clear all overlay boxes, including mate overlays and move overlays
        if hasattr(self, 'current_mate_overlays'):
            for box in self.current_mate_overlays:
                box.destroy()
            self.current_mate_overlays.clear()

        if hasattr(self, 'current_move_overlays'):
            for box in self.current_move_overlays:
                box.destroy()
            self.current_move_overlays.clear()

    def toggle_player_color(self):#Method to toggle the player's color
        # Function to toggle the player's color checkbox
        if self.player_color_var.get() == "White":
            self.player_color_var.set("Black")
        else:
            self.player_color_var.set("White")
        self.analyze_board()  # Update the board after toggling
   
    def on_debug_checkbox_change(self, *args):#Method to handle the debug checkbox change
        if not self.debug_checkbox_var.get():  # Check if the checkbox is unticked
            print("Debug Mode was turned off")
            self.reset_image()  # Clear the image when unticked

    def toggle_on_top(self):#Method to toggle the 'always on top' status
        # Toggles the 'always on top' status based on the checkbox state
        root.wm_attributes("-topmost", self.keep_on_top_var.get())

    def load_coordinates(self):#    Load the coordinates from the JSON file
        try:
            with open("chessboard_coordinates.json", "r") as file:
                data = json.load(file)
                self.start_x = data.get("start_x")
                self.start_y = data.get("start_y")
                self.end_x = data.get("end_x")
                self.end_y = data.get("end_y")
        except FileNotFoundError:
            pass

    def save_coordinates(self):#Save the coordinates to the JSON file
        data = {
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y
        }
        with open("chessboard_coordinates.json", "w") as file:
            json.dump(data, file)

    def load_svm_model(self):#Load the SVM model for chess piece detection
        # Load the pre-trained SVM model for chess piece detection
        model_path = r"C:\GitHubRepos\ChessBoardViewer\chess_piece_svm_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            print("Model file not found. Please train the SVM model first.")
            return None

    def select_area(self):  # Function to select the area of the chessboard
        # Hide the window to take a screenshot of full screen
        self.root.withdraw()
        time.sleep(0.5)  # Small delay to make sure the window is hidden

        # Take screenshot of the full screen
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # Create a new tkinter window to select the area
        selection_window = tk.Toplevel(self.root)
        selection_window.attributes("-fullscreen", True)
        selection_window.attributes("-alpha", 0.3)
        selection_window.configure(bg='black')

        selection_canvas = tk.Canvas(selection_window, cursor="cross", bg='black')
        selection_canvas.pack(fill="both", expand=True)

        # Create a small zoom window
        zoom_window = tk.Toplevel(selection_window)
        zoom_window.title("Zoom View")
        zoom_window.geometry("200x200")  # Fixed size for zoom window
        zoom_label = tk.Label(zoom_window)
        zoom_label.pack()

        self.rect = None

        def on_mouse_down(event):
            self.start_x, self.start_y = event.x, event.y
            self.rect = selection_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

        def on_mouse_move(event):
            if self.rect is not None:
                selection_canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

            # Update zoom window
            x, y = event.x, event.y
            zoom_size = 40  # Size of the area to zoom in on
            zoom_factor = 5  # Zoom factor

            # Calculate bounding box for zoom area
            left = max(0, x - zoom_size // 2)
            top = max(0, y - zoom_size // 2)
            right = min(screenshot.shape[1], x + zoom_size // 2)
            bottom = min(screenshot.shape[0], y + zoom_size // 2)

            # Crop and resize the zoom area
            zoom_area = screenshot[top:bottom, left:right]
            zoom_area = cv2.resize(zoom_area, (zoom_size * zoom_factor, zoom_size * zoom_factor), interpolation=cv2.INTER_LINEAR)
            zoom_area = cv2.cvtColor(zoom_area, cv2.COLOR_BGR2RGB)

            # Draw a box around the center pixel
            zoom_area = np.array(zoom_area)
            center_x = zoom_area.shape[1] // 2
            center_y = zoom_area.shape[0] // 2
            cv2.rectangle(zoom_area, (center_x - 2, center_y - 2), (center_x + 2, center_y + 2), (255, 0, 0), 1)

            zoom_image = Image.fromarray(zoom_area)
            zoom_photo = ImageTk.PhotoImage(zoom_image)
            zoom_label.config(image=zoom_photo)
            zoom_label.image = zoom_photo

        def on_mouse_up(event):
            self.end_x, self.end_y = event.x, event.y
            selection_window.destroy()
            zoom_window.destroy()
            # Show the main window again
            self.root.deiconify()
            # Save the selected coordinates
            self.save_coordinates()
            # Automatically analyze the board after selecting
            self.analyze_board()

        selection_canvas.bind("<ButtonPress-1>", on_mouse_down)
        selection_canvas.bind("<B1-Motion>", on_mouse_move)
        selection_canvas.bind("<ButtonRelease-1>", on_mouse_up)
        selection_window.bind("<Motion>", on_mouse_move)

    def calculate_mate(self):
        print("Calculating forced mate moves...")

    def update_evaluation_display(self, evaluation_data, fen):#Update the evaluation display
        if evaluation_data["type"] == "cp":
            # Centipawn value indicates positional advantage
            eval_value = evaluation_data["value"]

            # Update progress bar (value from 0 to 100)
            # Assume a scale of -1000 to +1000 centipawns for evaluation
            eval_score_normalized = max(-1000, min(1000, eval_value))  # Clamp value between -1000 and +1000
            eval_percentage = (eval_score_normalized + 1000) / 20  # Convert to percentage (0 to 100)

            self.evaluation_bar["value"] = eval_percentage

            # Update evaluation label
            if eval_value > 0:
                self.eval_label.config(text=f"Evaluation: +{eval_value} (White is better)")
            elif eval_value < 0:
                self.eval_label.config(text=f"Evaluation: {eval_value} (Black is better)")
            else:
                self.eval_label.config(text="Evaluation: Equal")

        elif evaluation_data["type"] == "mate":
            # Mate in X moves found
            mate_in = evaluation_data["value"]     

            # Update evaluation bar to show extreme value for mate
            self.evaluation_bar["value"] = 100 if mate_in > 0 else 0

            # Update evaluation label
            self.eval_label.config(text=f"Mate in {mate_in} moves")

            if self.show_mate_var.get():
                # Get the forced mate moves
                forced_mate_moves = self.get_forced_mate_moves(fen)
                if forced_mate_moves:
                    print("Forced mate moves:", forced_mate_moves)
                    self.highlight_forced_mate(forced_mate_moves)

                else:
                    print("No forced mate moves found.")

    def reset_image(self): #Reset the image on the GUI
        # Clear the image by setting it to None
        self.image_label.configure(image=None)
        self.image_label.imgtk = None  # Remove reference to the image
        placeholder_img = Image.new("RGB", (1, 1), color=(255, 255, 255))  # White 1x1 pixel
        imgtk = ImageTk.PhotoImage(image=placeholder_img)
        
        # Update the image_label with the 1x1 placeholder
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)
        # Resize the window to fit the new contents
        self.root.geometry("")  # Lets Tkinter recalculate the window size based on content

    def test_highlight_mate_moves(self):
        # Example FEN for testing
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # Example moves for testing (7 moves)
        test_moves = ["a1a2", "b1b2", "c1c2", "d1d2", "e1e2", "f1f2", "g1g2", "h1h2", "a3,a4"]
        self.highlight_forced_mate(test_moves)
        
#Stockfish methods     
    def initialize_stockfish(self):  # Initialize Stockfish
        try:
            # Ensure you're retrieving the slider's value as an integer
            skill_level = self.fish_skill_slider.get() if hasattr(self.fish_skill_slider, 'get') else self.fish_skill_slider
            self.stockfish = Stockfish(self.stockfish_path)

            # Set skill level and Stockfish parameters to use more threads
            self.stockfish.set_skill_level(skill_level)
            self.stockfish.update_engine_parameters({"Threads": 6})  # Adjust as needed based on your system

            print(f"Stockfish initialized. Skill level: {skill_level}, Threads: 6")
        except Exception as e:
            print(f"Failed to initialize Stockfish: {e}")
            self.stockfish = None

    def Restart_stockfish(self):#   Restart Stockfish
        del self.stockfish  # Python should handle cleanup
        self.initialize_stockfish()

    def is_legal_fen(self, fen: str) -> bool:#Check if the FEN is legal
        try:
            board = chess.Board(fen)
            if board.is_valid() and board.is_check() is not None: # Check if the board is valid
                print("Valid FEN.")
                return True
            else:
                print("Invalid FEN.")
                if self.reanalyze_var.get():
                    threading.Timer(1.0, self.analyze_board).start()  # Delay analyze_board by 1 second
                return False
        except ValueError:
            print("Invalid FEN syntax.")
            if self.reanalyze_var.get():
                threading.Timer(1.0, self.analyze_board).start()  # Delay analyze_board by 1 second
            return False

    def get_evaluation_data(self, fen: str):#Get the evaluation data from Stockfish
        if self.stockfish is None:
            print("Stockfish is not initialized. Reinitializing...")
            self.initialize_stockfish()
            if self.stockfish is None:
                print("Failed to reinitialize Stockfish.")
                return

        try:
            # Set the FEN position
           # self.stockfish.set_fen_position(fen)

            # Get the evaluation from Stockfish
            evaluation = self.stockfish.get_evaluation()

            # Update GUI safely
            self.root.after(0, self.update_evaluation_display, evaluation, fen)

        except Exception as e:
            print(f"Stockfish process crashed: {e}. Reinitializing...")
            self.initialize_stockfish()

    def get_best_move(self, fen): #Get the best move from Stockfish
            if self.stockfish is None:
                print("Stockfish is not initialized. Reinitializing...")
                self.initialize_stockfish()
                if self.stockfish is None:
                    return None, None
            
            try:
                # Set the FEN position
                self.stockfish.set_fen_position(fen)
                
                # Get the best move from Stockfish
                think_time = self.think_time_slider.get()
                best_move = self.stockfish.get_best_move_time(think_time)
                
            except Exception as e:
                print(f"Stockfish process crashed: {e}. Reinitializing...")
                self.initialize_stockfish()
                if self.stockfish is None:
                    return None, None
                self.stockfish.set_fen_position(fen)
                best_move = self.stockfish.get_best_move()
            if best_move is None:
                return None, None
            

            # **Promotion Check**:
            if len(best_move) == 5:
                best_move = best_move[:4]  # Strip off the promotion character (e.g., "q")

            # Split the move into starting and ending coordinates
            start_pos = best_move[:2]
            end_pos = best_move[2:]
            
            return start_pos, end_pos
    
    def get_forced_mate_moves(self, fen: str) -> list:
        # Load the chess engine
        engine_path = r"C:\GitHubRepos\ChessBoardViewer\stockfish\stockfish-windows-x86-64-avx2.exe"
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        board = chess.Board(fen)
        mate_moves = []

        try:
            # Get Stockfish analysis with a lower depth to detect a mate in 3
            info = engine.analyse(board, chess.engine.Limit(depth=12))  # Lower depth for faster computation
            if info.get("score") and info["score"].is_mate() and 1 <= abs(info["score"].relative.mate()) <= 10:
                # Mate in 3 detected, extract the principal variation (PV)
                pv_moves = info.get("pv", [])
                for i in range(0, len(pv_moves), 2):  # Get only your moves (skipping opponent's moves)
                    mate_moves.append(pv_moves[i].uci())

        finally:
            engine.quit()

        return mate_moves

    def ping_stockfish(self):
        try:
            # Use get_parameters to check if Stockfish is responsive
            parameters = self.stockfish.get_parameters()
            if parameters:
                print("Stockfish is responsive.")
                return True
            else:
                print("No response from Stockfish.")
                return False
        except Exception as e:
            print(f"Error while pinging Stockfish: {e}")
            return False

#Main Methods
    def analyze_board(self): #Main function to analyze the board and call draw methods


        if self.start_x is None or self.start_y is None or self.end_x is None or self.end_y is None:
            print("Please select an area first.")
            return

        if not self.svm_model:
            print("SVM model not loaded. Cannot analyze the board.")
            return

        think_time_ms = self.think_time_slider.get()  # Get the think time from the slider
        think_time_seconds = think_time_ms / 1000.0  # Convert milliseconds to seconds
        self.deny_hotkeys_for(think_time_seconds)  # Deny hotkeys for the same amount of time as the think time

        # Hide overlays before taking a screenshot
        self.hide_overlays()
        
        # Take a screenshot of the selected area
        screenshot = ImageGrab.grab(bbox=(self.start_x, self.start_y, self.end_x, self.end_y))
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        # Show overlays after taking the screenshot
        self.show_overlays()
        
        # Assume the selected area is a perfect square chessboard
        board_size = 8
        height, width, _ = screenshot.shape
        cell_width = width // board_size
        cell_height = height // board_size
        
        # Draw the chessboard grid and identify pieces
        self.screenshot_with_grid = screenshot.copy()
        cells = []
        board_position = [['' for _ in range(board_size)] for _ in range(board_size)]
        for i in range(board_size):
            for j in range(board_size):
                top_left = (j * cell_width, i * cell_height)
                bottom_right = ((j + 1) * cell_width, (i + 1) * cell_height)
                cell = screenshot[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                cells.append(cell)
                
        # Perform color analysis using mean brightness to determine piece colors
        colors = self.determine_piece_colors(cells)
        
        # Define the chessboard coordinates based on player color
        board_coordinates = self.get_board_coordinates()

        for idx, (i, j) in enumerate([(i, j) for i in range(board_size) for j in range(board_size)]):
            top_left = (j * cell_width, i * cell_height)
            bottom_right = ((j + 1) * cell_width, (i + 1) * cell_height)
            cell = cells[idx]
            
            # Identify the piece in the cell using HOG and SVM
            piece_name, confidence = self.identify_piece_with_svm(cell)
            if (piece_name and confidence > 0.8):  # Only consider predictions with high confidence
                # Add piece color label at the bottom of the cell if not identified as empty
                piece_color = colors[idx] if piece_name.lower() != "empty" else ""
                if piece_name.lower() != "empty":
                    color_label_y = bottom_right[1] - 10
                    color = (0, 0, 0) if piece_color == 'Black' else (255, 255, 255)
                    cv2.putText(self.screenshot_with_grid, piece_color, (top_left[0] + 5, color_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    
                    # Update board position
                    piece_fen = 'n' if piece_name.lower() == 'knight' and piece_color == 'Black' else 'N' if piece_name.lower() == 'knight' and piece_color == 'White' else piece_name[0].lower() if piece_color == 'Black' else piece_name[0].upper()
                    board_position[i][j] = piece_fen
                
                # Use the piece color to adjust the final prediction if applicable
                if piece_color == "Black" and "White" in piece_name:
                    continue  # Skip if prediction does not match color
                if piece_color == "White" and "Black" in piece_name:
                    continue  # Skip if prediction does not match color
                
                cv2.putText(self.screenshot_with_grid, piece_name, (top_left[0] + 5, top_left[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Add coordinates in the middle of each cell with orange color
            coord = board_coordinates[i][j]
            coord_x = top_left[0] + cell_width // 2 - 10
            coord_y = top_left[1] + cell_height // 2 + 5
            cv2.putText(self.screenshot_with_grid, coord, (coord_x, coord_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
            
            # Draw the grid
            cv2.rectangle(self.screenshot_with_grid, top_left, bottom_right, (255, 0, 0), 1)
        
        # Generate FEN notation
        castling_rights = self.check_castling_rights(board_position)
        fen_rows = []
        for row in board_position:
            empty_count = 0
            fen_row = ''
            for cell in row:
                if cell == '':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        fen = '/'.join(fen_rows) + (' b' if self.player_color_var.get() == 'Black' else ' w') + f' {castling_rights} - 0 1'
        
        # Flip the FEN if the player is black to adjust the board perspective
        if self.player_color_var.get() == 'Black':
            fen_rows = fen.split(' ')[0].split('/')
            fen_rows = [row[::-1] for row in reversed(fen_rows)]
            fen = '/'.join(fen_rows) + (' b' if self.player_color_var.get() == 'Black' else ' w') + f' {castling_rights} - 0 1'
          
        print(f"FEN: {fen[:37]}")

        # Display the updated board with grid
        self.display_image(self.screenshot_with_grid)

        if self.is_legal_fen(fen):
            if not self.ping_stockfish():
                self.Restart_stockfish()
            else:
                # Use a separate thread to get the best move to avoid blocking the GUI
                threading.Thread(target=self.draw_moves_and_print, args=(fen,)).start()
                think_time = self.think_time_slider.get()
                self.root.after(think_time, lambda: None)  # Non-blocking pause for the length of the think time (in ms) 
                if self.eval_box_var.get():
                    threading.Thread(target=self.get_evaluation_data, args=(fen,)).start()    

    def check_castling_rights(self, board_position): #Check if castling is possible to add to FEN
        white_castling_rights = {'K': True, 'Q': True}
        black_castling_rights = {'k': True, 'q': True}
        
        # Automatically determine if kings have been in check based on board state
        # Check if white king or rooks have moved or if there are pieces in the way for castling
        if board_position[7][4] != 'K' or board_position[7][7] != 'R' or board_position[7][5] or board_position[7][6]:
            white_castling_rights['K'] = False

        if board_position[7][4] != 'K' or board_position[7][0] != 'R' or board_position[7][1] or board_position[7][2] or board_position[7][3]:
            white_castling_rights['Q'] = False

        # Check if black king or rooks have moved or if there are pieces in the way for castling
        if board_position[0][4] != 'k' or board_position[0][7] != 'r' or board_position[0][5] or board_position[0][6]:
            black_castling_rights['k'] = False

        if board_position[0][4] != 'k' or board_position[0][0] != 'r' or board_position[0][1] or board_position[0][2] or board_position[0][3]:
            black_castling_rights['q'] = False

        # Build castling rights string
        castling_rights = ""
        castling_rights += 'K' if white_castling_rights['K'] else ''
        castling_rights += 'Q' if white_castling_rights['Q'] else ''
        castling_rights += 'k' if black_castling_rights['k'] else ''
        castling_rights += 'q' if black_castling_rights['q'] else ''
        return castling_rights if castling_rights else '-'




    def determine_piece_colors(self, cells): #Determine the color of the pieces
        colors = []
        # Get the current threshold value from the slider
        brightness_threshold = self.threshold_value.get()
        for cell in cells:
            # Convert the cell to grayscale
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            # Calculate the average brightness of the cell
            avg_brightness = np.mean(cell_gray)
            
            # Classify based on brightness threshold
            if avg_brightness > brightness_threshold:
                colors.append("White")
            else:
                colors.append("Black")
    
        return colors

    def draw_moves_and_print(self, fen): #Get the best move and draw it on the GUI
        start, end = self.get_best_move(fen)
        if start is None or end is None:
            print("No move available.")
            return

        print(f"The best move is from {start} to {end}.")
        
        # Now that we have the move, we need to update the GUI.
        # Tkinter GUI updates must be run on the main thread.
        self.root.after(0, self.highlight_over_GUI, start, end)
        self.root.after(0, self.highlight_over_screen, start, end)

    def highlight_over_screen(self, start, end): #Highlight the best move on the screen
        # Assume the selected area is a perfect square chessboard
        board_size = 8
        cell_width = (self.end_x - self.start_x) // board_size
        cell_height = (self.end_y - self.start_y) // board_size

        # Define the chessboard coordinates based on player color
        board_coordinates = self.get_board_coordinates()

        start_x, start_y = self.get_cell_coordinates(start, board_coordinates, cell_width, cell_height)
        end_x, end_y = self.get_cell_coordinates(end, board_coordinates, cell_width, cell_height)
        
        if start_x is not None and end_x is not None:
            # Create a list for move overlays if it doesn't exist
            if not hasattr(self, 'current_move_overlays'):
                self.current_move_overlays = []

            # Destroy existing move overlay boxes before creating new ones
            for overlay in self.current_move_overlays:
                overlay.destroy()
            self.current_move_overlays = []

            # Convert local coordinates to screen coordinates
            screen_start_x = self.start_x + start_x
            screen_start_y = self.start_y + start_y
            screen_end_x = self.start_x + end_x
            screen_end_y = self.start_y + end_y

            # Create a transparent tkinter window for drawing the highlight box for the start position
            overlay = tk.Toplevel(self.root)
            overlay.attributes("-transparentcolor", "magenta")
            overlay.attributes("-topmost", True)
            overlay.geometry(f"{cell_width}x{cell_height}+{screen_start_x}+{screen_start_y}")
            overlay.overrideredirect(True)

            canvas = tk.Canvas(overlay, width=cell_width, height=cell_height, bg='magenta', highlightthickness=0)
            canvas.pack()
            canvas.create_rectangle(0, 0, cell_width, cell_height, outline='green', width=5)

            # Add the overlay to the list of current move overlays
            self.current_move_overlays.append(overlay)

            # Create a transparent tkinter window for drawing the highlight box for the end position
            overlay_end = tk.Toplevel(self.root)
            overlay_end.attributes("-transparentcolor", "magenta")
            overlay_end.attributes("-topmost", True)
            overlay_end.geometry(f"{cell_width}x{cell_height}+{screen_end_x}+{screen_end_y}")
            overlay_end.overrideredirect(True)

            canvas_end = tk.Canvas(overlay_end, width=cell_width, height=cell_height, bg='magenta', highlightthickness=0)
            canvas_end.pack()
            canvas_end.create_rectangle(0, 0, cell_width, cell_height, outline='red', width=5)

            # Add the overlay to the list of current move overlays
            self.current_move_overlays.append(overlay_end)

    def highlight_over_GUI(self, start, end): #Highlight the best move on the GUi
        # Assume the selected area is a perfect square chessboard
        board_size = 8
        cell_width = (self.end_x - self.start_x) // board_size
        cell_height = (self.end_y - self.start_y) // board_size

        # Define the chessboard coordinates based on player color
        board_coordinates = self.get_board_coordinates()

        start_x, start_y = self.get_cell_coordinates(start, board_coordinates, cell_width, cell_height)
        end_x, end_y = self.get_cell_coordinates(end, board_coordinates, cell_width, cell_height)
        
        if start_x is not None and end_x is not None:
            # Draw the best move on the original screenshot with grid
            cv2.rectangle(self.screenshot_with_grid, (start_x, start_y), (start_x + cell_width, start_y + cell_height), (0, 255, 0), 3)
            cv2.rectangle(self.screenshot_with_grid, (end_x, end_y), (end_x + cell_width, end_y + cell_height), (0, 0, 255), 3)
            
            # Display the result in the GUI
            self.display_image(self.screenshot_with_grid)

    def draw_screen_highlight_box(self, x, y, cell_width, cell_height, color, text=None, position='center'):
        # Create a transparent tkinter window for drawing the highlight box
        overlay = tk.Toplevel(self.root)
        overlay.attributes("-transparentcolor", "magenta")
        overlay.attributes("-topmost", True)
        overlay.geometry(f"{cell_width}x{cell_height}+{x}+{y}")
        overlay.overrideredirect(True)

        canvas = tk.Canvas(overlay, width=cell_width, height=cell_height, bg='magenta', highlightthickness=0)
        canvas.pack()
        canvas.create_rectangle(0, 0, cell_width, cell_height, outline=color, width=5)

        # Draw the number if text is provided
        if text is not None:
            corners = {
                'nw': (10, 10),
                'ne': (cell_width - 10, 10),
                'sw': (10, cell_height - 10),
                'se': (cell_width - 10, cell_height - 10),
                'center': (cell_width // 2, cell_height // 2)
            }
            pos_x, pos_y = corners[position]
            canvas.create_text(pos_x, pos_y, text=text, anchor=position, fill=color, font=('Helvetica', 24))

        # Add the overlay to the list of current mate overlays
        self.current_mate_overlays.append(overlay)
        
    def highlight_forced_mate(self, mate_moves): 

        # Destroy existing mate overlay boxes before creating new ones
        if hasattr(self, 'current_mate_overlays'):
            for overlay in self.current_mate_overlays:
                overlay.destroy()
            self.current_mate_overlays = []

        # Limit to the first five moves
        mate_moves = mate_moves[:5]

        # Get board size and cell dimensions
        board_size = 8
        cell_width = (self.end_x - self.start_x) // board_size
        cell_height = (self.end_y - self.start_y) // board_size

        # Get the board coordinates
        board_coordinates = self.get_board_coordinates()

        # Create a list for mate overlays if it doesn't exist
        if not hasattr(self, 'current_mate_overlays'):
            self.current_mate_overlays = []

        # Highlight each move
        colors = ['yellow', 'blue', 'orange', 'purple', 'cyan']
        corners = ['nw', 'ne', 'sw', 'se', 'center']
        for idx, move in enumerate(mate_moves):
            start = move[:2]
            end = move[2:]

            start_x, start_y = self.get_cell_coordinates(start, board_coordinates, cell_width, cell_height)
            end_x, end_y = self.get_cell_coordinates(end, board_coordinates, cell_width, cell_height)

            if start_x is not None and end_x is not None:
                # Convert local coordinates to screen coordinates
                screen_start_x = self.start_x + start_x
                screen_start_y = self.start_y + start_y
                screen_end_x = self.start_x + end_x
                screen_end_y = self.start_y + end_y

                # Create overlay for the start position
                self.draw_screen_highlight_box(screen_start_x, screen_start_y, cell_width, cell_height,
                                    colors[idx % len(colors)], str(idx + 1), corners[idx % len(corners)])

                # Create overlay for the end position
                self.draw_screen_highlight_box(screen_end_x, screen_end_y, cell_width, cell_height,
                                    'red', str(idx + 1), corners[idx % len(corners)])



    def get_cell_coordinates(self, coord, board_coordinates, cell_width, cell_height): #Get the coordinates of the cell
        for i, row in enumerate(board_coordinates):
            for j, board_coord in enumerate(row):
                if board_coord == coord:
                    return j * cell_width, i * cell_height
        return None, None

    def identify_piece_with_svm(self, cell): #Identify the piece using SVM
        # Resize cell to a fixed size
        cell_resized = cv2.resize(cell, (64, 64))
        
        # Convert to grayscale
        cell_gray = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        features, _ = hog(cell_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        
        # Predict the piece using the SVM model
        prediction = self.svm_model.decision_function([features])
        confidence = max(prediction[0]) if len(prediction[0]) > 0 else 0
        predicted_class = self.svm_model.classes_[np.argmax(prediction)] if confidence > 0 else None
        return predicted_class, confidence

    def display_image(self, img): #Display the image on the GUI
        # Only display the image if the checkbox is enabled
        if self.debug_checkbox_var.get():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            # Resize the image to a set size
            img = img.resize((600, 600))  # Set desired width and height
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

    def get_board_coordinates(self): #Get the board coordinates based on player color
        player_color = self.player_color_var.get()
        board_size = 8
        if (player_color == "White"):
            return [[f'{chr(97 + j)}{8 - i}' for j in range(board_size)] for i in range(board_size)]
        else:
            return [[f'{chr(97 + (board_size - 1 - j))}{i + 1}' for j in range(board_size)] for i in range(board_size)]
        
    def on_close(self): # Function to close the GUI and unhook all keyboard events
        keyboard.unhook_all()  # Stop all keyboard hooks
        self.root.destroy()         # Close the GUI window

    def auto_detect_board(self):
        # Hide the window to take a screenshot of full screen
        self.root.withdraw()
        time.sleep(0.2)  # Small delay to make sure the window is hidden

        # Take screenshot of the full screen
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # Convert the screenshot to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find the chessboard
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which should be the chessboard
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Set the coordinates
            self.start_x, self.start_y, self.end_x, self.end_y = x, y, x + w, y + h

            print(f"Chessboard detected at coordinates: ({self.start_x}, {self.start_y}), ({self.end_x}, {self.end_y})")

            # Save the selected coordinates
            self.save_coordinates()
            # Automatically analyze the board after selecting
            self.analyze_board()
        else:
            print("No chessboard detected. Please try again.")

        # Show the main window again
        self.root.deiconify()

    def monitor_board_changes(self):
        if self.reanalyze_var.get():
            print("Recapture enabled. Monitoring board changes...")
            # Hide overlays before taking a screenshot
            self.hide_overlays()
            
            # Take a screenshot of the selected area
            screenshot = ImageGrab.grab(bbox=(self.start_x, self.start_y, self.end_x, self.end_y))
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            
            # Show overlays after taking the screenshot
            self.show_overlays()

            # Compare with the last capture
            if self.last_capture is not None:
                if not np.array_equal(screenshot, self.last_capture):
                    self.analyze_board()
                else:
                    pass
            else:
                print("No previous capture to compare with.")

            # Update the last capture
            self.last_capture = screenshot

        # Schedule the next capture
        capture_interval = self.capture_interval_slider.get()
        self.root.after(capture_interval, self.monitor_board_changes)

    def hide_overlays(self):
        if hasattr(self, 'current_mate_overlays'):
            for overlay in self.current_mate_overlays:
                overlay.withdraw()
        if hasattr(self, 'current_move_overlays'):
            for overlay in self.current_move_overlays:
                overlay.withdraw()

    def show_overlays(self):
        if hasattr(self, 'current_mate_overlays'):
            for overlay in self.current_mate_overlays:
                overlay.deiconify()
        if hasattr(self, 'current_move_overlays'):
            for overlay in self.current_move_overlays:
                overlay.deiconify()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChessBoardDetector(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Bind the close event to the on_close function
    root.mainloop()
    #keyboard.wait()  # Keep the program running to listen for hotkeys
