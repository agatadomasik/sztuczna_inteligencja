from connect4 import Connect4
from exceptions import AgentException
import copy


class AlphaBetaAgent:
    def __init__(self, my_token='x', max_depth=5):
        self.my_token = my_token
        self.max_depth = max_depth

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        _, best_move = self.minimax(copy.deepcopy(connect4), self.max_depth, True, float("-inf"), float("inf"))
        return best_move

    def minimax(self, s, d, x, alpha, beta):
        if s.game_over or d == 0:
            if s.wins == self.my_token:
                return 1, None
            elif s.wins is None:
                return self.heuristics(s), None
                #return 0, None
            else:
                return -1, None

        drops = s.possible_drops()
        if x:
            max_v = float('-inf')
            best_drop = None
            for drop in drops:
                s_tmp = copy.deepcopy(s)
                s_tmp.drop_token(drop)
                v, _ = self.minimax(s_tmp, d - 1, False, alpha, beta)
                if v > max_v:
                    max_v = v
                    best_drop = drop
                alpha = max(alpha, max_v)
                if max_v >= beta:
                    break
            return max_v, best_drop
        else:
            min_v = float('inf')
            best_drop = None
            for drop in drops:
                s_tmp = copy.deepcopy(s)
                s_tmp.drop_token(drop)
                v, _ = self.minimax(s_tmp, d - 1, True, alpha, beta)
                if v < min_v:
                    min_v = v
                    best_drop = drop
                beta = min(beta, min_v)
                if min_v <= alpha:
                    break
            return min_v, best_drop

    def heuristics(self, connect4):
        score = 0
        other_token = 'x' if self.my_token == 'o' else 'o'
        fours = connect4.iter_fours()
        for four in fours:
            count_my_token = four.count(self.my_token)
            count_other_token = four.count(other_token)

            if count_my_token == 3:
                score += 100
            elif count_other_token == 3:
                score -= 100

            elif count_my_token == 2:
                score += 10
            elif count_other_token == 2:
                score -= 10

            elif count_my_token == 1:
                score += 1
            elif count_other_token == 1:
                score -= 1

        return score

