you should use the grafitti board and the default pieces if you can
you must select if you are black or white

press detect board for the program to load where the game board is.

press analyze board to show the best move for the currently loaded stockfish

if you change the stockfish skill level you must also restart stockfish.

to use the full auto mode you need to select a small area of the timer that does not contain a number or the timer. this box is going to look for color changes when its your turn the box will highlight

HOTKEYS
        keyboard.add_hotkey('a', lambda: self.analyze_board_if_ready())
        keyboard.add_hotkey('s', lambda: self.toggle_player_color())
        keyboard.add_hotkey('d', lambda: self.get_best_premove())
        keyboard.add_hotkey('f', lambda: self.toggle_reanalyze())
