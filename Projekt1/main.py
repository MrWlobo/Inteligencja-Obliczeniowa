from easyAI import TwoPlayerGame
import random

# Convert D7 to (3,6) and back...
to_string = lambda move: " ".join(
    ["ABCDEFGHIJ"[move[i][0]] + str(move[i][1] + 1) for i in (0, 1)]
)
to_tuple = lambda s: ("ABCDEFGHIJ".index(s[0]), int(s[1:]) - 1)

class Hexapawn(TwoPlayerGame):
    """
    A nice game whose rules are explained here:
    http://fr.wikipedia.org/wiki/Hexapawn
    """

    def __init__(self, players, size=(4, 4)):
        self.size = M, N = size
        p = [[(i, j) for j in range(N)] for i in [0, M - 1]]

        for i, d, goal, pawns in [(0, 1, M - 1, p[0]), (1, -1, 0, p[1])]:
            players[i].direction = d
            players[i].goal_line = goal
            players[i].pawns = pawns

        self.players = players
        self.current_player = 1
        self.removed_pawns =  []

    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        d = self.player.direction
        for i, j in self.player.pawns:
            if (i + d, j) not in opponent_pawns:
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
            # owner is the index (0 or 1) of the player who owned the captured pawn
            owner = 0 if self.opponent is self.players[0] else 1
            # store (owner, column) so resurrected pawn has the same column as original
            self.removed_pawns.append((owner, move[1][1]))
            self.opponent.pawns.remove(move[1])
        
        if self.removed_pawns and random.random() < 0.3:
            owner, col = random.choice(self.removed_pawns)
            # compute owner's home row (player 0 starts at row 0, player 1 at M-1)
            home_row = 0 if self.players[owner].direction == 1 else (self.size[0] - 1)
            start_pos = (home_row, col)
            # per your rule, don't check occupancy here (you said it cannot happen)
            self.players[owner].pawns.append(start_pos)
            try:
                self.removed_pawns.remove((owner, col))
            except ValueError:
                pass


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
    from easyAI import AI_Player, Human_Player, Negamax

    scoring = lambda game: -100 if game.lose() else 0
    ai = Negamax(10, scoring)
    game = Hexapawn([AI_Player(ai), AI_Player(ai)])
    game.play()
    print("player %d wins after %d turns " % (game.opponent_index, game.nmove))
