from time import time

import chess
from random import shuffle, randrange
from core import LexWolfCore
from bitBoard import bitBoard


class MinmaxLexWolf(LexWolfCore):
    def __init__(self, center_bonus=0.1, control_bonus=0.1, king_bonus=0.2, check_bonus=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_bonus = center_bonus
        self.control_bonus = control_bonus
        self.king_bonus = king_bonus
        self.check_bonus = check_bonus
        self.start_time = time()

    def evaluate(self, board, bitBrd):

        # Initial score
        score = 0

        # Material and positional score
        bitBrd.setList(board)
        score = bitBrd.getEval()

        # Checkmate and stalemate
        if board.is_checkmate():
            if board.turn:
                score -= 100000  # Black wins
            else:
                score += 100000  # White wins
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            score = 0  # Draw

        # Add control bonus
        #if self.control_bonus:
        #    white_control = self.count_controlled_squares(board, chess.WHITE)
        #    black_control = self.count_controlled_squares(board, chess.BLACK)
        #    score += self.control_bonus * sum(white_control.values())
        #    score -= self.control_bonus * sum(black_control.values())

        # Add check bonus
        if self.check_bonus and board.is_check():
            score += self.check_bonus

        return score

    def minimax(self, board, depth, alpha, beta, is_maximizing, bitBrd):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board, bitBrd)

        if is_maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False, bitBrd)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, max_eval)
                if alpha >= beta:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True, bitBrd)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Alpha cut-off                
            return min_eval

    def find_optimal_move(self, board=chess.Board(), bitBrd=bitBoard()):
        turn = board.turn
        self.start_time = time()
        legal_moves = list(board.legal_moves)
        shuffle(legal_moves)
        best_move = legal_moves[0]
        best_value = float('-inf') if turn == chess.WHITE else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            if time() - self.start_time > self.max_thinking_time:
                break
            board.push(move)
            self.combinations_count = 1
            board_value = self.minimax(board, self.max_depth - 1, alpha, beta, not turn,bitBrd)
            board.pop()
            r = randrange(2)

            if board.turn == chess.WHITE:
                if board_value > best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move
                    alpha = max(alpha, best_value)  # Update alpha
            else:
                if board_value < best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move
                    beta = min(beta, best_value)  # Update beta

        return best_move
        
    def safe_move(self, previous_move, new_move, board):
        if new_move in board.legal_moves:
            return new_move
        else:
            return previous_move
