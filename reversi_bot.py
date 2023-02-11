import numpy as np
import random as rand
import reversi
import copy

# Swap initially from 1/2 to 1/-1,
# Change flipper to match
# Run algorithm on a child for opposite
# Storing as nodes in a tree
# run a minmax
# run a/b
# modify heuristics


class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num
        self.square_vals = np.array([[99,-8,8,6,6,8,-8,99],[-8,-24,-4,-3,-3,-4,-24,-8],[8,-4,7,4,4,7,-4,8],[6,-3,4,0,0,4,-3,6],
                            [6,-3,4,0,0,4,-3,6],[8,-4,7,4,4,7,-4,8],[-8,-24,-4,-3,-3,-4,-24,-8],[99,-8,8,6,6,8,-8,99]])

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
        print('###################################### NEW MOVE ###############')
        valid_moves = state.get_valid_moves()
        self.state = state
        self.turn = state.turn
        self.opponent = 2 if state.turn == 1 else 1

        curr_fin_val = -1000
        fin_move = valid_moves[0]

        for move in valid_moves:
            child, tile_count, op_tile_count = self.create_child(state.board, move)
            mv_val = self.eval_qual(child)
            if mv_val > curr_fin_val:
                fin_move = move

        n_state = reversi.ReversiGameState(child, self.opponent)
        n_val_moves = n_state.get_valid_moves()

        return fin_move

    def create_child(self, board, move, ):
        state_copy = copy.deepcopy(board)
        state_copy[move[0]][move[1]] = self.turn
        tile_count = 0
        opp_tile_count = 0
        for x in range(len(state_copy)):
            for y in range(len(state_copy[0])):
                if state_copy[x][y] == self.opponent:
                    state_copy[x][y] = -1
                    opp_tile_count += 1
                if state_copy[x][y] == self.turn:
                    state_copy[x][y] = 1
                    tile_count += 1

        self.changeColors(move[0], move[1], state_copy)
        return state_copy, tile_count, opp_tile_count

    def board_heatmap(self, state):
        val = state * self.square_vals
        fin_val = sum(sum(col) for col in val)

        return fin_val

    def eval_qual(self, state):
        val = self.board_heatmap(state)

        fin_val = val
        return fin_val







    def checkDirection(self, row, col, incx, incy, state):
        sequence = [0 for _ in range(8)]

        seqLen = 0

        for i in range(1, 8):
            r = row + incy * i
            c = col + incx * i

            if (r < 0) or (r > 7) or (c < 0) or (c > 7):
                break

            sequence[i] = state[r][c]
            seqLen += 1

        count = 0
        for i in range(seqLen): # TURNS = 1 or 2
            if self.turn == 1:
                if sequence[i] == -1:
                    count += 1
                else:
                    if (sequence[i] == 1) and (count > 0):
                        count = 20
                        break
            else:
                if sequence[i] == 1:
                    count += 1
                else:
                    if (sequence[i] == -1) and (count > 0):
                        count = 20
                    break

        if count > 10:
            if self.turn == 1:
                i = 1
                r = row + incy * i
                c = col + incx * i
                while state[r][c] == -1:
                    state[r][c] = 1
                    i += 1
                    r = row + incy * i
                    c = col + incx * i

            else:
                i = 1
                r = row + incy * i
                c = col + incx * i
                while state[r][c] == 1:
                    state[r][c] = 2
                    i += 1
                    r = row + incy * i
                    c = col + incx * i

        return state

    def changeColors(self, row, col, state):
        for incx in range(-1, 2):
            for incy in range(-1, 2):
                if (incx == 0) and (incy == 0):
                    continue

                state = self.checkDirection(row, col, incx, incy, state)

        return state
