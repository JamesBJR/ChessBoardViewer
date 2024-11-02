import os
import time
import json
import cv2
import numpy as np
import pyautogui
import chess

class ChessBoardDetector:
    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.templates = self.load_templates()

    def load_templates(self):
        templates = {}
        pieces = ['wp', 'bp', 'wr', 'br', 'wn', 'bn', 'wb', 'bb', 'wq', 'bq', 'wk', 'bk']
        for piece in pieces:
            templates[piece] = cv2.imread(f'templates/{piece}.png', 0)
        return templates

    def take_screenshot(self):
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def detect_chessboard(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h

    def detect_pieces(self, chessboard_img):
        board_size = 8
        cell_width = chessboard_img.shape[1] // board_size
        cell_height = chessboard_img.shape[0] // board_size
        board_position = [['' for _ in range(board_size)] for _ in range(board_size)]
        for i in range(board_size):
            for j in range(board_size):
                cell = chessboard_img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                piece = self.match_template(cell)
                if piece:
                    board_position[i][j] = piece
        return board_position

    def match_template(self, cell):
        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        best_match = None
        best_val = float('inf')
        for piece, template in self.templates.items():
            res = cv2.matchTemplate(cell_gray, template, cv2.TM_SQDIFF)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            if min_val < best_val:
                best_val = min_val
                best_match = piece
        return best_match

    def generate_fen(self, board_position):
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
        return '/'.join(fen_rows) + ' w - - 0 1'

    def run(self):
        screenshot = self.take_screenshot()
        edges = self.preprocess_image(screenshot)
        x, y, w, h = self.detect_chessboard(edges)
        chessboard_img = screenshot[y:y+h, x:x+w]
        board_position = self.detect_pieces(chessboard_img)
        fen = self.generate_fen(board_position)
        print(f"FEN: {fen}")

if __name__ == "__main__":
    detector = ChessBoardDetector()
    detector.run()
