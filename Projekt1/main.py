from easyAI import TwoPlayerGame
import random
import copy
import time

# Convert D7 to (3,6) and back...
to_string = lambda move: " ".join(
    ["ABCDEFGHIJ"[move[i][0]] + str(move[i][1] + 1) for i in (0, 1)]
)
to_tuple = lambda s: ("ABCDEFGHIJ".index(s[0]), int(s[1:]) - 1)

class Expectiminimax:
    def __init__(self, depth, scoring):
        self.depth = depth
        self.scoring = scoring

    def __call__(self, game):
        """Metoda wywoływana przez AI_Player, aby znaleźć najlepszy ruch."""
        best_move = None
        max_val = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in game.possible_moves():
            game_copy = copy.deepcopy(game)
            game_copy.play_move(move)
            val = self._expecti(game_copy, self.depth - 1, alpha, beta, False)
            if val > max_val:
                max_val = val
                best_move = move
            alpha = max(alpha, val)
        return best_move

    def _expecti(self, game, depth, alpha, beta, maximizing):
        if depth == 0 or game.is_over():
            score = self.scoring(game)
            return score if maximizing else -score

        if maximizing:
            v = -float('inf')
            for move in game.possible_moves():
                game_copy = copy.deepcopy(game)
                game_copy.play_move(move)
                v = max(v, self._expecti(game_copy, depth - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if beta <= alpha: break
            return v
        else:
            v = float('inf')
            for move in game.possible_moves():
                game_copy = copy.deepcopy(game)
                game_copy.play_move(move)
                res = self._expecti(game_copy, depth - 1, alpha, beta, True)
                v = min(v, res)
                beta = min(beta, v)
                if beta <= alpha: break
            return v

class Hexapawn(TwoPlayerGame):
    """
    A nice game whose rules are explained here:
    http://fr.wikipedia.org/wiki/Hexapawn
    """
    def __init__(self, players, starting_player, size=(4, 4)):
        self.size = M, N = size
        p = [[(i, j) for j in range(N)] for i in [0, M - 1]]

        for i, d, goal, pawns in [(0, 1, M - 1, p[0]), (1, -1, 0, p[1])]:
            players[i].direction = d
            players[i].goal_line = goal
            players[i].pawns = list(pawns)

        self.players = players
        self.current_player = starting_player
        self.removed_pawns = []
    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        d = self.player.direction
        for i, j in self.player.pawns:
            if (i + d, j) not in opponent_pawns and (i + d, j) not in self.player.pawns:
                moves.append(((i, j), (i + d, j)))
            if (i + d, j + 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j + 1)))
            if (i + d, j - 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j - 1)))

        return list(map(to_string, [(i, j) for i, j in moves]))

    def make_move(self, move):
        move = list(map(to_tuple, move.split(" ")))
        ind = self.player.pawns.index(move[0])
        self.player.pawns[ind] = move[1]

        if move[1] in self.opponent.pawns:
            owner = 0 if self.opponent is self.players[0] else 1
            self.removed_pawns.append((owner, move[1][1]))
            self.opponent.pawns.remove(move[1])
        
        if self.removed_pawns and random.random() < 0.1:
            idx = random.randint(0, len(self.removed_pawns) - 1)
            owner, col = self.removed_pawns.pop(idx)
            home_row = 0 if self.players[owner].direction == 1 else (self.size[0] - 1)
            start_pos = (home_row, col)
            all_occupied = self.players[0].pawns + self.players[1].pawns
            if start_pos not in all_occupied:
                self.players[owner].pawns.append(start_pos)


    def lose(self):
        return any([i == self.opponent.goal_line for i, j in self.opponent.pawns]) or (
            self.possible_moves() == []
        )

    def is_over(self):
        return self.lose()

    def show(self):
        f = (
            lambda x: "1"
            if x in self.players[0].pawns
            else ("2" if x in self.players[1].pawns else ".")
        )
        print(
            "\n".join(
                [
                    " ".join([f((i, j)) for j in range(self.size[1])])
                    for i in range(self.size[0])
                ]
            )
        )


if __name__ == "__main__":
    games_count = 100
    depth = 3
    use_alpha_beta_pruning = False
    use_expecti = True

    from easyAI import AI_Player, Human_Player, Negamax

    scoring = lambda game: -100 if game.lose() else 0

    if use_expecti:
        ai = Expectiminimax(depth, scoring)
    else:
        ai = Negamax(depth, scoring, use_alpha_beta_pruning=use_alpha_beta_pruning)

    player1 = AI_Player(ai)
    player2 = AI_Player(ai)

    player1_wins = 0
    player2_wins = 0
    starting_player = 1

    average_decision_times = []
    for _ in range(games_count):
        game = Hexapawn([player1, player2], starting_player)
        start = time.time()
        game.play()
        end = time.time()
        print("player %d wins after %d turns " % (game.opponent_index, game.nmove))
        print(f"Average move time: {(end - start) / game.nmove}")
        average_decision_times.append((end - start) / game.nmove)

        if game.opponent_index == 1:
            player1_wins += 1
        else:
            player2_wins += 1

        starting_player = 1 if starting_player == 2 else 2


    print("------------------------------")
    print(f"Player 1 Wins: {player1_wins}")
    print(f"Player 2 Wins: {player2_wins}")
    print(f"Average decision time over all games: {sum(average_decision_times) / len(average_decision_times)}")
    print(f"Depth: {depth}")
    print(f"Alpha-Beta pruning used: {use_alpha_beta_pruning}")
