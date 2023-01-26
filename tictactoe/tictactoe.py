"""
Tic Tac Toe Player
"""

import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = 0
    o = 0
    for row in board:
        for column in row:
            if column == X:
                x += 1
            elif column == O:
                o += 1
    if x == o:
        return X
    elif x > o:
        return O
    elif x < o:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    action_set = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != "X" and board[i][j] != "O":
                action_set.add((i, j))
    return action_set


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i = action[0]
    j = action[1]
    board_mod = copy.deepcopy(board)
    if action in actions(board_mod):
        del board_mod[i][j]
        if player(board_mod) == X:
            board_mod[i].insert(j, X)
        elif player(board_mod) == O:
            board_mod[i].insert(j, O)
        return board_mod
    else:
        raise Exception


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Rows
    if board[0][0] == X and board[0][1] == X and board[0][2] == X:
        return X
    elif board[0][0] == O and board[0][1] == O and board[0][2] == O:
        return O
    elif board[1][0] == X and board[1][1] == X and board[1][2] == X:
        return X
    elif board[1][0] == O and board[1][1] == O and board[1][2] == O:
        return O
    elif board[2][0] == X and board[2][1] == X and board[2][2] == X:
        return X
    elif board[2][0] == O and board[2][1] == O and board[2][2] == O:
        return O

    # Columns
    elif board[0][0] == X and board[1][0] == X and board[2][0] == X:
        return X
    elif board[0][0] == O and board[1][0] == O and board[2][0] == O:
        return O
    elif board[0][1] == X and board[1][1] == X and board[2][1] == X:
        return X
    elif board[0][1] == O and board[1][1] == O and board[2][1] == O:
        return O
    elif board[0][2] == X and board[1][2] == X and board[2][2] == X:
        return X
    elif board[0][2] == O and board[1][2] == O and board[2][2] == O:
        return O

    # Diagonals
    elif board[0][0] == X and board[1][1] == X and board[2][2] == X:
        return X
    elif board[0][0] == O and board[1][1] == O and board[2][2] == O:
        return O
    elif board[0][2] == X and board[1][1] == X and board[2][0] == X:
        return X
    elif board[0][2] == O and board[1][1] == O and board[2][0] == O:
        return O
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) == X or winner(board) == O:
        return True

    else:
        for row in board:
            for cell in row:
                if cell == EMPTY:
                    return False

        else:
            return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def max_value(state):
        if terminal(state):
            return utility(state)
        else:
            value = -99999
            for action in actions(state):
                value = max(value, min_value(result(state, action)))
            return value

    def min_value(state):
        if terminal(state):
            return utility(state)
        else:
            value = 99999
            for action in actions(state):
                value = min(value, max_value(result(state, action)))
            return value

    if terminal(board):
        return None

    choices = []
    if player(board) == X:
        for a in actions(board):
            board_value = min_value(result(board, a))
            choices.append((board_value, a))
        choices.sort(reverse=True)

    else:
        for a in actions(board):
            board_value = max_value(result(board, a))
            choices.append((board_value, a))
        choices.sort()

    return choices[0][1]
