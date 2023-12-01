"""
Tic Tac Toe Player
"""

import copy
import math

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
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    
    if x_count == o_count:
        return X
    
    if x_count > o_count:
        return O



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))

    return possible_actions
    


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    
    if board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid action")

    new_board = copy.deepcopy(board)  #cela permet de creer une copy de board car sans cela board change aussi quand new_board change
    new_board[action[0]][action[1]] = player(board)
    return new_board
    
    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    for i in range(3):    #horizontalement
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        
    for j in range(3):      #verticalement
        if board[0][j] == board[1][j] == board[2][j] != EMPTY:
            return board[0][j]
        
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[1][1]
    
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[1][1]
    
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) == X or winner(board) == O:
        return True
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                return False
    
    return True
    
    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    
    if winner(board) == O:
        return -1
    
    else:
        return 0
    
    raise NotImplementedError


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    plus_actions = set()
    moyen_actions = set()
    worst_actions = set()
    
    if player(board) == X:
        
        for action in actions(board):
            if min_value(result(board, action)) == 1:
                plus_actions.add(action)
            if min_value(result(board, action)) == 0:
                moyen_actions.add(action)
            else:
                worst_actions.add(action)
        
        if plus_actions:
            for action in plus_actions:
                return action
        elif moyen_actions:
            for action in moyen_actions:
                return action
        else:
            for action in worst_actions:
                return action
        
        
    
    if player(board) == O:
         
        for action in actions(board):
            if min_value(result(board, action)) == -1:
                plus_actions.add(action)
            if min_value(result(board, action)) == 0:
                moyen_actions.add(action)
            else:
                worst_actions.add(action)
        
        if plus_actions:
            for action in plus_actions:
                return action
        elif moyen_actions:
            for action in moyen_actions:
                return action
        else:
            for action in worst_actions:
                return action
            


def max_value(board):
    if terminal(board):
        return utility(board)
    
    v = -math.inf
    
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v

def min_value(board):
    if terminal(board):
        return utility(board)
    
    v = math.inf
    
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v