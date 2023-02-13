import numpy as np
import random as rand
import reversi
import copy
import multiprocessing

# Swap initially from 1/2 to 1/-1,          $
# Change flipper to match                   $
# Run algorithm on a child for opposite     $
# Storing as nodes in a tree                $
# run a minmax
# run a/b
# modify heuristics
# do we recompile the tree each time? or flesh out the tree a bit,      $
# $then continue fleshing based on what the opp played                  $


class ReversiBot:
    def __init__(self, move_num):
        self.our_turn = move_num
        self.opponent_turn = 2 if self.our_turn == 1 else 1
        self.init_state = np.array([[0 for i in range(8)] for j in range(8)])
        self.game_tree = GameTree(GameNode(reversi.ReversiGameState(self.init_state, self.our_turn)))
        self.ab = AlphaBeta(self.game_tree)


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
        is_in_tree = self.game_tree.find_grandchild_node(state.board)
        if not is_in_tree:
            self.game_tree.curr = GameNode(reversi.ReversiGameState(state.board, self.our_turn))

        # Run algorithm through game tree to pick best child
        self.game_tree.generate_3_generations()
        # self.game_tree.generate_3_parallel()


        # Pick Best
        child = self.ab.alpha_beta_search(self.game_tree.curr)

        # valid_moves = state.get_valid_moves()
        # print(len(valid_moves))
        return child.parent_move

class FindChildParallel:
    def __call__(self, node):
        node.create_children()

class FindGrandParallel:
    def __call__(self, node):
        node.create_children()
        for great_grandchild in node.children:
            great_grandchild.create_children()
            # for greatgreat in great_grandchild.children:
            #     greatgreat.create_children()
            #     for greatgreatgreat in greatgreat.children:
            #         greatgreatgreat.create_children()

class FindGreatGrandParallel:
    def __call__(self, nodes):
        nodes.create_children()
        for child in nodes.children:
            child.create_children()
            for grandchild in child.children:
                grandchild.create_children()
                # for great_grandchild in grandchild.children:
                #     great_grandchild.create_children()


class GameTree:
    def __init__(self, root):
        self.root = root
        self.curr = root
        self.generate_grandchildren()

        self.proc = FindGrandParallel()
        # self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())


    def generate_grandchildren(self):
        self.curr.create_children()
        for child in self.curr.children:
            child.create_children()

    def generate_3_generations(self):
        self.curr.create_children()
        for child in self.curr.children:
            child.create_children()
            for grandchild in child.children:
                grandchild.create_children()
                for great_grandchild in grandchild.children:
                    great_grandchild.create_children()

    def generate_3_parallel(self):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # with multiprocessing.Pool(processes=1) as pool:
            self.curr.create_children()
            for child in self.curr.children:
                child.create_children()
                for grandchild in child.children:
                    grandchild.create_children()
                    pool.map(self.proc, grandchild.children)

    # def generate_3_parallel(self):
    #     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         self.curr.create_children()
    #         for child in self.curr.children:
    #             child.create_children()
    #             pool.map(self.proc, child.children)

    # def generate_3_parallel(self):
    #     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #         self.curr.create_children()
    #         for child in self.curr.children:
    #             child.create_children()
    #             pool.map(self.proc, child.children)


    def find_grandchild_node(self, board):
        # Find which GameTree's grandchild matches current game state
        for child in self.curr.children:
            for grandchild in child.children:
                if (grandchild.state.board == board).min():
                    self.curr = grandchild
                    return True
        print('false')
        return False



class GameNode:
    def __init__(self, state, parent_move=None, value=0):
        self.state = state
        self.children = []
        self.valid_moves = self.getValidMoves(state.turn)
        self.parent_move = parent_move
        self.opponent_turn = -1 if self.state.turn == 1 else 1
        self.value = value
        self.square_vals = np.array(
            [[99, -8, 8, 6, 6, 8, -8, 99], [-8, -24, -4, -3, -3, -4, -24, -8], [8, -4, 7, 4, 4, 7, -4, 8],
             [6, -3, 4, 0, 0, 4, -3, 6],
             [6, -3, 4, 0, 0, 4, -3, 6], [8, -4, 7, 4, 4, 7, -4, 8], [-8, -24, -4, -3, -3, -4, -24, -8],
             [99, -8, 8, 6, 6, 8, -8, 99]])

        self.name = "root" if parent_move is None else str(parent_move)


    def create_children(self):
        if len(self.children) != 0:
            return
        for move in self.valid_moves:
            self.children.append(self.create_child(move))

    def board_heatmap(self, state):
        val = state * self.square_vals
        fin_val = sum(sum(col) for col in val)

        return fin_val

    def tile_count(self, state):
        s = 0
        for row in state:
            s += sum(row)
        return s


    def eval_quality(self, child):
        state = child.state.board
        hm = self.board_heatmap(state)
        # tile count
        tc = self.tile_count(state)
        # move count
        mc = len(child.getValidMoves())

        fin_val = hm
        fin_val += -2*tc
        fin_val += mc * 2
        return fin_val

    def create_child(self, move):
        state_copy = copy.deepcopy(self.state.board)
        state_copy[move[0]][move[1]] = self.state.turn
        # tile_count = 0
        # opp_tile_count = 0

        self.changeColors(move[0], move[1], state_copy)
        child = GameNode(reversi.ReversiGameState(state_copy, self.opponent_turn), move)

        value = self.eval_quality(child)

        return GameNode(reversi.ReversiGameState(state_copy, self.opponent_turn), move, value)

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
            if sequence[i] == -1:
                count = count + 1
            else:
                if (sequence[i] == 1) and (count > 0):
                    return True
                break


        return False



    def getValidMoves(self, me = 1):
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
        for child in successors:
            value = self.min_value(child, best_val, beta)
            if value > best_val:
                best_val = value
                best_state = child
        print("AlphaBeta: Utility Value of Root Node: = " + str(best_val))
        # print("AlphaBeta: Best State is: " + best_state.name)
        return best_state

    def max_value(self, node, alpha, beta):
        # print("AlphaBeta->MAX: Visited Node :: " + node.name)
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = -infinity

        for child in node.children:
            value = max(value, self.min_value(child, alpha, beta))
            if value >= beta:
                # idx = node.children.index(child)
                # print("cutting ", len(node.children), "to", idx, "from max")
                # node.children = node.children[:idx+1]
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, node, alpha, beta):
        # print("AlphaBeta->MIN: Visited Node :: " + node.name)
        if self.isTerminal(node):
            return self.getUtility(node)
        infinity = float('inf')
        value = infinity

        for child in node.children:
            value = min(value, self.max_value(child, alpha, beta))
            if value <= alpha:
                # idx = node.children.index(child)
                # print("cutting ", len(node.children), "to", idx, "from min")
                # node.children = node.children[:idx+1]
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
        return node.value