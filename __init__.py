import chess
import chess.svg
from random import shuffle
from core import DummyLexWolf, LexWolfCore
from bitBoard import bitBoard
from IPython.display import display, SVG


class Game():
    """
    Handles the chess games
    """
    def __init__(self, p1_is_human=True, p2_is_human=False, AIwhite=DummyLexWolf(), AIblack=DummyLexWolf(),
                 max_moves=None, verbose=1, silence=False):
        self.p1_is_human = p1_is_human
        self.p2_is_human = p2_is_human
        self.AIwhite = AIwhite  # type LexWolfCore
        self.AIblack = AIblack
        self.board = chess.Board()
        self.bitBrd = bitBoard()
        self.move_memory = []
        self.max_moves = max_moves  # stops the game if > max_moves, inactive if set on "None"
        self.move_count = 0
        self.result = 0  # -1 --> black, 0 --> draw, 1 --> white
        self.result_cause = 'Checkmate'
        self.create_opening_library()
        self.verbose = verbose
        self.silence = silence

        self.start()  # starts the game

    def create_opening_library(self, path="lexwolf/openings.csv", output_path="lexwolf/openings_fen.csv"):
        """
        The idea is to build a csv database from another csv file. The source csv file contains lists of opening moves
        in the column 'moves_list', e.g. "['1.e4', 'Nf6', '2.e5', 'Nd5', '3.d4', 'd6', '4.Bc4']" and a winning percentage
        in the column 'Player Win %'. The input are those two columns.
        The output is a csv file that, for each fen position, tells what is the best move to play according to the
        opening database. The output csv file will have the following columns: 'fen', 'best_move', 'winning_percentage'.
        The 'best_move' format shall be in the format 'e2e4' instead of '1.e4', so that it can be used by the AI.
        """
        # Check if the output csv file already exists
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Please remove it or choose another file name.")
            return

        # Read the source csv file
        openings_lists_csv = pd.read_csv(path)
        # Initialize the output DataFrame
        openings_fen_csv = pd.DataFrame(columns=['fen', 'best_move', 'winning_percentage'])

        # Temporary dictionary to hold FEN positions and their moves with winning percentages
        fen_dict = {}

        for _, row in openings_lists_csv.iterrows():
            moves_list = eval(row['moves_list'])
            win_percentage = row['Player Win %']
            board = chess.Board()

            for move in moves_list:
                # Convert move to move object
                try:
                    move_obj = board.parse_san(move)
                except ValueError:
                    # If move can't be parsed, it might be an invalid move or not applicable in the current board state
                    break

                board.push(move_obj)

                # Convert to FEN
                fen = board.fen()

                # Determine move in long algebraic notation
                move_lan = board.san(move_obj)

                # Update or add the move and win percentage to the dictionary
                if fen not in fen_dict:
                    fen_dict[fen] = []
                fen_dict[fen].append((move_lan, win_percentage))

                # Prepare for next iteration
                board = chess.Board(fen)

        # Process the dictionary to find the best move for each FEN
        for fen, moves_win_percs in fen_dict.items():
            best_move, winning_percentage = max(moves_win_percs, key=lambda x: x[1])
            # Convert best move to long algebraic notation if necessary
            board = chess.Board(fen)
            best_move_lan = board.san(board.parse_san(best_move))
            # Append to the DataFrame
            openings_fen_csv = openings_fen_csv.append({'fen': fen, 'best_move': best_move_lan, 'winning_percentage': winning_percentage}, ignore_index=True)

        # Write the DataFrame to a new CSV file
        openings_fen_csv.to_csv(output_path, index=False)
        print(f"Opening library created and saved to {output_path}.")


    def AImove(self, AI):
        self.show_board(size=500)
        next_move = AI.find_optimal_move(self.board) if AI.incrEval == False else AI.find_optimal_move_incremental(self.board)
        if next_move in self.board.legal_moves:
            self.load_move_in_memory(next_move)
            self.play_move(next_move)
            self.verbose_message(self.generate_AI_reaction())
        else:
            print(self.board)
            print(next_move)
            raise ValueError("The AI just played an illegal move.")

    def check_endgame(self):
        if self.board.is_checkmate():
            return 1
        elif self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves()\
                or self.board.is_insufficient_material() or self.board.can_claim_draw():
            return -1
        else:
            return 0

    def human_move(self):
        self.show_board(size=500)
        valid = False
        while not valid:
            uci = input("\nYour turn: ")
            try:
                move = chess.Move.from_uci(uci)
                assert(move in self.board.legal_moves)
                self.load_move_in_memory(move)
                self.play_move(move)
                valid = True
                self.verbose_message("\nYour move is valid. Waiting for opponent...")
            except:
                self.verbose_message(f"This move is not legal. Legal moves are: {list(self.board.legal_moves)}")

    def generate_AI_reaction(self):
        reactions = ["The AI played its move.",
                     "The AI played its move. Fear, puny human!",
                     "The AI has established a strategy to crush you.",
                     "AI: hmmm... let's see how you will handle that.",
                     "AI: I played my move. And I didn't ask to GPT."]
        reactions += ["The AI played its move."] * 10
        shuffle(reactions)
        return reactions[0]

    def load_move_in_memory(self, move):
        self.move_memory.append(move)

    def message(self, mes):
        if not self.silence:
            print(mes)

    def play_move(self, move):
        self.board.push(move)
        self.bitBrd.setList(self.board)
        if self.verbose and not self.silence:
            print("Value of next move:", self.bitBrd.getEval())

    def start(self):
        self.message("\n\nTHE GAME HAS STARTED. GOOD LUCK!\n")
        while True:
            self.move_count += 1
            # White move
            if self.p1_is_human:
                self.human_move()
            else:
                self.AImove(self.AIwhite)
                self.show_board()
            if self.check_endgame() == 1:
                # White win
                self.message(f"Checkmate at move {self.move_count}")
                self.result = 1
                break
            elif self.check_endgame() == -1:
                # Draw
                self.result = 0
                break

            # Black move
            if self.p2_is_human:
                self.human_move()
            else:
                self.AImove(self.AIblack)
            if self.check_endgame() == 1:
                # Black win
                self.message(f"Checkmate at move {self.move_count}")
                self.result = -1
                break
            elif self.check_endgame() == -1:
                # Stalemate --> draw
                self.result = 0
                break

            if self.max_moves is not None and self.move_count > self.max_moves:
                self.verbose_message("EXCEEDED MAX ALLOWED MOVES FOR THIS GAME.")
                break


        if self.result == 0:
            if self.board.is_insufficient_material():
                self.message(f"Draw for insufficient material at move {self.move_count}")
                self.result_cause = "Insufficient material"
            elif self.board.is_stalemate():
                self.message(f"Draw for stalemate at move {self.move_count}")
                self.result_cause = "Stalemate"
            elif self.board.is_fivefold_repetition():
                self.message(f"Draw for fivefold repetition at move {self.move_count}")
                self.result_cause = "Fivefold repetition"
            elif self.board.can_claim_draw():
                self.message(f"Draw claimed or threefold repetition at move {self.move_count}")
                self.result_cause = "Threefold repetition"
            elif self.board.is_seventyfive_moves():
                self.message(f"Draw according to the 75-moves law at move {self.move_count}")
                self.result_cause = "75 moves"


        self.verbose_message("\nYou can visualize the full gameplay by calling 'Game.show_game()'.")
        self.message(f"Result: {self.result}")
        self.message("\nGAME OVER.")

    def show_board(self, svg_board=True, size=200):
        if svg_board and not self.silence:
            svg = chess.svg.board(self.board, size=size)  # Generate SVG for the current board
            display(SVG(svg))  # Display the SVG in Jupyter Notebook
        else:
            self.verbose_message(self.board)  # Fallback to verbose message if SVG not desired

    def show_game(self):
        self.board = chess.Board()
        for move in self.move_memory:
            self.play_move(move)
            self.message("")
            self.show_board()

    def verbose_message(self, message):
        if self.verbose:
            self.message(message)
