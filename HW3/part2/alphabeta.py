from player import Player
from board import Board
import math


class AlphaBetaPlayer(Player):
    def __init__(self, player_number, board):
        super().__init__(player_number, board)
        self.max_depth = 4

    def get_next_move(self):
        return self.best_move(self.board, self.max_depth)

    def best_move(self, board, depth):

        alpha = -math.inf
        beta = math.inf
        best_value = -math.inf
        best_move = None
        score0, score1 = self.board.get_scores()

        for i in range(self.board.get_n()):
            for j in range(self.board.get_n()):
                if self.board.is_move_valid(self.player_number, i, j):
                    self.board.start_imagination()
                    count = self.board.imagine_placing_piece(self.player_number, i, j)
                    value = self.minimax(board, alpha, beta, 1, self.opponent_number)
                    if value > best_value:
                        best_value = value
                        best_move = (i, j)

        return best_move

    def minimax(self, board, alpha, beta, depth, player_number):
        exist = False
        for i in range(board.get_n()):
            e = False
            for j in range(board.get_n()):
                if board.is_imaginary_move_valid(player_number, i, j):
                    e = True
                    break
            if e:
                exist = True
                break

        if not exist or depth == self.max_depth:
            count0 = 0
            count1 = 0
            for i in range(board.get_n()):
                for j in range(board.get_n()):
                    if board.imaginary_board_grid[i][j] == 0:
                        count0 += 1
                    elif board.imaginary_board_grid[i][j] == 1:
                        count1 += 1
            if self.player_number == 0:
                return count0 - count1
            else:
                return count1 - count0

        if player_number == self.player_number:
            best_value = -math.inf

            for i in range(board.get_n()):
                for j in range(board.get_n()):
                    if board.is_imaginary_move_valid(player_number, i, j):
                        b1 = Board()
                        b1.imaginary_board_grid = board.imaginary_board_grid
                        b1.imagine_placing_piece(player_number, i, j)
                        value = self.minimax(b1, alpha, beta, depth + 1, self.opponent_number)
                        best_value = max(best_value, value)
                        alpha = max(alpha, best_value)
                        if beta <= alpha:
                            break
            return best_value
        else:
            best_value = math.inf

            for i in range(board.get_n()):
                for j in range(board.get_n()):
                    if board.is_imaginary_move_valid(player_number, i, j):
                        b1 = Board()
                        b1.imaginary_board_grid = board.imaginary_board_grid
                        b1.imagine_placing_piece(player_number, i, j)
                        value = self.minimax(b1, alpha, beta, depth + 1, self.player_number)
                        best_value = min(best_value, value)
                        beta = min(beta, best_value)
                        if beta <= alpha:
                            break

            return best_value
