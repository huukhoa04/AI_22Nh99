class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        # Tạo một list biểu diễn bảng 3x3
    def print_board(self):
        row1 = '|'.join(self.board[0:3])
        row2 = '|'.join(self.board[3:6])
        row3 = '|'.join(self.board[6:9])
        print(row1)
        print('-' * 5)
        print(row2)
        print('-' * 5)
        print(row3)
    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']
        # Trả về list các ô trống
    def is_winner(self, symbol):
        # Kiểm tra trạng thái thắng của một người chơi
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for condition in win_conditions:
            if all(self.board[i] == symbol for i in condition):
                return True
        return False
    def minimax(self, maximizing_player, depth):
        if self.is_winner('O'):
            # Nếu máy tính thắng trò chơi, trả về điểm 10 - depth
            return 10 - depth
        if self.is_winner('X'):
            # Nếu người chơi thắng trò chơi, trả về điểm depth - 10
            return depth - 10
        if ' ' not in self.board:
            # Trò chơi hòa, trả về 0
            return 0
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.available_moves():
                self.board[move] = 'O'
                eval = self.minimax(False, depth + 1)
                self.board[move] = ' '
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.available_moves():
                self.board[move] = 'X'
                eval = self.minimax(True, depth + 1)
                self.board[move] = ' '
                min_eval = min(min_eval, eval)
            return min_eval
    def find_best_move(self):
        best_move = -1
        best_eval = float('-inf')
        for move in self.available_moves():
            self.board[move] = 'O'
            eval = self.minimax(False, 0)
            self.board[move] = ' '
            if eval > best_eval:
                best_eval = eval
                best_move = move
        return best_move
game = TicTacToe()
game.print_board()
while ' ' in game.board:
    user_move = int(input('Enter your move (0-8): '))
    game.board[user_move] = 'X'
    game.print_board()
    if game.is_winner('X'):
        print('You win!')
        break
    if ' ' not in game.board:
        print('Draw!')
        break
    ai_move = game.find_best_move()
    game.board[ai_move] = 'O'
    game.print_board()
    if game.is_winner('O'):
        print('AI wins!')
        break
    if ' ' not in game.board:
        print('Draw!')
        break

