import numpy as np
import random as rand
import reversi
import copy


# Swap initially from 1/2 to 1/-1,          $
# Change flipper to match                   $
# Run algorithm on a child for opposite     $
# Storing as nodes in a tree                $
# run a minmax
# run a/b
# modify heuristics
# do we recompile the tree each time? or flesh out the tree a bit, then continue fleshing based on what the opp played  $


class ReversiBot:
    def __init__(self, move_num):
        self.our_turn = move_num
        self.opponent_turn = 2 if self.our_turn == 1 else 1
        self.init_state = np.array([[0 for i in range(8)] for j in range(8)])
        self.game_tree = GameTree(GameNode(reversi.ReversiGameState(self.init_state, self.our_turn), None))

    def make_move(self, state):
        '''
        This is the only function that needs to be implemented for the lab!
        The bot should take a game state and return a move.

        The parameter "state" is of type ReversiGameState and has two useful
        member variables. The first is "board", which is an 8x8 numpy array
        of 0s, 1s, and 2s. If a spot has a 0 that means it is unoccupied. If
        there is a 1 that means the spot has one of player 1's stones. If
        there is a 2 on the spot that means that spot has one of player 2's
        stones. The other useful member variable is "turn", which is 1 if it's
        player 1's turn and 2 if it's player 2's turn.

        ReversiGameState objects have a nice method called get_valid_moves.
        When you invoke it on a ReversiGameState object a list of valid
        moves for that state is returned in the form of a list of tuples.

        Move should be a tuple (row, col) of the move you want the bot to make.
        '''
        # Parse board into max/min format

        for x in range(len(state.board)):
            for y in range(len(state.board[0])):
                if state.board[x][y] == self.our_turn:
                    state.board[x][y] = 1
                elif state.board[x][y] == self.opponent_turn:
                    state.board[x][y] = -1

        # Returns true if all equal, false if any are not
        x = self.game_tree.find_grandchild_node(state.board)

        # Run algorithm through game tree to pick best child
        self.game_tree.generate_grandchildren()

        valid_moves = state.get_valid_moves()
        print(len(valid_moves))
        return valid_moves[0]

class GameTree:
    def __init__(self, root):
        self.root = root
        self.curr = root
        self.square_vals = np.array([[99,-8,8,6,6,8,-8,99],[-8,-24,-4,-3,-3,-4,-24,-8],[8,-4,7,4,4,7,-4,8],[6,-3,4,0,0,4,-3,6],
                            [6,-3,4,0,0,4,-3,6],[8,-4,7,4,4,7,-4,8],[-8,-24,-4,-3,-3,-4,-24,-8],[99,-8,8,6,6,8,-8,99]])
        self.generate_grandchildren()


    def board_heatmap(self, state):
        val = state * self.square_vals
        fin_val = sum(sum(col) for col in val)

        return fin_val

    def eval_qual(self, state):
        val = self.board_heatmap(state)

        fin_val = val
        return fin_val

    def generate_grandchildren(self):
        self.curr.create_children()
        for child in self.curr.children:
            child.create_children()

    def find_grandchild_node(self, board):
        # Find which GameTree's grandchild matches current game state
        for child in self.curr.children:
            for grandchild in child.children:
                if (grandchild.state.board == board).min():
                    self.curr = grandchild
                    return True
        return False


class GameNode:
    def __init__(self, state, parent_move):
        self.state = state
        self.children = []
        self.valid_moves = self.getValidMoves(state.turn)
        self.parent_move = parent_move
        self.opponent_turn = -1 if self.state.turn == 1 else 1



    def create_children(self):
        for move in self.valid_moves:
            child_board = self.create_child(move)
            self.children.append(GameNode(reversi.ReversiGameState(child_board, self.opponent_turn), move))


    def create_child(self, move):
        state_copy = copy.deepcopy(self.state.board)
        state_copy[move[0]][move[1]] = self.state.turn
        tile_count = 0
        opp_tile_count = 0

        # for x in range(len(state_copy)):
        #     for y in range(len(state_copy[0])):
        #         if state_copy[x][y] == self.opponent:
        #             state_copy[x][y] = -1
        #             opp_tile_count += 1
        #         if state_copy[x][y] == self.state.turn:
        #             state_copy[x][y] = 1
        #             tile_count += 1

        self.changeColors(move[0], move[1], state_copy)
        return state_copy

    def couldBe(self, row, col):
        for incx in range(-1, 2):
            for incy in range(-1, 2):
                if (incx == 0) and (incy == 0):
                    continue

                if self.checkDirection(row, col, incx, incy):
                    return True

        return False

    def changeColors(self, row, col, state):
        for incx in range(-1, 2):
            for incy in range(-1, 2):
                if (incx == 0) and (incy == 0):
                    continue

                if self.checkDirection(row, col, incx, incy):
                    state = self.flipTiles(row, col, incx, incy, state)

        return state

    def flipTiles(self, row, col, incx, incy, state):
        i = 1
        r = row + incy * i
        c = col + incx * i
        while state[r][c] == self.opponent_turn:
            state[r][c] = self.state.turn
            i += 1
            r = row + incy * i
            c = col + incx * i


        return state

    def checkDirection(self, row, col, incx, incy):
        sequence = []
        for i in range(1, 8):
            r = row + incy * i
            c = col + incx * i

            if ((r < 0) or (r > 7) or (c < 0) or (c > 7)):
                break

            sequence.append(self.state.board[r][c])

        count = 0
        for i in range(len(sequence)):
            if self.state.turn == 1:
                if sequence[i] == -1:
                    count = count + 1
                else:
                    if (sequence[i] == 1) and (count > 0):
                        return True
                    break
            else:
                if sequence[i] == 1:
                    count = count + 1
                else:
                    if (sequence[i] == -1) and (count > 0):
                        return True
                    break

        return False

    def getValidMoves(self, me):
        validMoves = []

        if 0 in np.array(self.state.board)[3:5, 3:5]:
            for row in range(3, 5):
                for col in range(3, 5):
                    if self.state.board[row][col] == 0:
                        validMoves.append((row, col))
        else:
            for i in range(8):
                for j in range(8):
                    if self.state.board[i][j] == 0:
                        if self.couldBe(i, j):
                            validMoves.append([i, j])

        return validMoves

class AlphaBeta:
    # print names of all nodes visited during search
    def __init__(self, game_tree):
        self.game_tree = game_tree  # GameTree
        self.root = game_tree.root  # GameNode
        return

    def alpha_beta_search(self, node):
        infinity = float('inf')
        best_val = -infinity
        beta = infinity

        successors = self.getSuccessors(node)
        best_state = None
        for state in successors:
            value = self.min_value(state, best_val, beta)
            if value > best_val:
                best_val = value
                best_state = state
        print("AlphaBeta: Utility Value of Root Node: = " + str(best_val))
        print("AlphaBeta: Best State is: " + best_state.Name)
        return best_state

    def max_value(self, node, alpha, beta):
        print("AlphaBeta->MAX: Visited Node :: " + node.Name)
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = -infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = max(value, self.min_value(state, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha, beta):
        print("AlphaBeta->MIN: Visited Node :: " + node.Name)
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = infinity

        successors = self.getSuccessors(node)
        for state in successors:
            value = min(value, self.max_value(state, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    # successor states in a game tree are the child nodes...
    def getSuccessors(self, node):
        assert node is not None
        return node.children

    # return true if the node has NO children (successor states)
    # return false if the node has children (successor states)
    def isTerminal(self, node):
        assert node is not None
        return len(node.children) == 0

    def getUtility(self, node):
        assert node is not None
        return node.valu