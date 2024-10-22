import random
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from player_divercite import PlayerDivercite
from seahorse.game.game_layout.board import Piece

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses the Minimax algorithm with alpha-beta pruning.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "AlphaBetaPlayer"):
        super().__init__(piece_type, name)


    def check_almost_divercite_city(self, state, city_pos, threshold) -> bool:
        # Get the color of the piece at the given position, if any
        neighbors = state.get_neighbours(city_pos[0], city_pos[1])
        colors = {"B" : 0, "R" : 0, "Y" : 0, "G" : 0}
        if state.piece_type_match("C", city_pos):
            for n in neighbors.values():
                if n[0] != "EMPTY":
                    type = n[0].get_type()[0]
                    colors[type] += 1
            return len(colors) >= threshold and colors["R"] < 2 and colors["G"] < 2 and colors["B"] < 2 and colors["Y"] < 2
        else :
            return False
    
    def check_almost_divercite_res(self, state, res_pos, threshold):
        almost = False
        neighbors = state.get_neighbours(res_pos[0], res_pos[1])
        for n in neighbors.values():
            almost = almost or self.check_almost_divercite_city(state, n[1], threshold)
        return almost
    
    def check_almost_divercite(self, state, pos, threshold):
        if state.piece_type_match("C", pos):
            self.check_almost_divercite_city(state, pos, threshold)
        else:
            self.check_almost_divercite_res(state, pos, threshold)
    
    def is_quiescent(self, state: GameState):
        board = state.get_rep().get_env()
        for pos in board:
            if self.check_almost_divercite_city(state, pos, 3):
                return False
        return True

    def alpha_beta_minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
        if depth == 0 or state.is_done():
            return self.evaluate_state(state), None

        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            actions = state.get_possible_light_actions()

            # Ne trie que si le nombre d'actions est assez grand
            if len(actions) > 5:
                actions = sorted(actions, key=lambda a: self.evaluate_state(state.apply_action(a)), reverse=True)

            for action in actions:
                next_state = state.apply_action(action)
                eval, _ = self.alpha_beta_minimax(next_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                    alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Coupure
            return max_eval, best_action  # Return value and best action
        else:
            min_eval = float('inf')
            best_action = None
            actions = state.get_possible_light_actions()

            if len(actions) > 5:
                actions = sorted(actions, key=lambda a: self.evaluate_state(state.apply_action(a)))

            for action in actions:
                next_state = state.apply_action(action)
                eval, _ = self.alpha_beta_minimax(next_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Coupure
            return min_eval, best_action  # Return value and best action

    def greedy(self, state):
        possible_actions = state.generate_possible_heavy_actions()
        best_action = next(possible_actions)
        best_score = best_action.get_next_game_state().scores[self.get_id()]
        for action in possible_actions:
            state = action.get_next_game_state()
            score = state.scores[self.get_id()]
            if score > best_score:
                best_action = action
        return best_action

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Use the minimax algorithm with alpha-beta pruning to choose the best action.
        """
        if current_state.get_step() < 3:
            return self.greedy(current_state)
        else:
            players = current_state.players
            players_id = [p.get_id() for p in players]
            dic_player_pieces = current_state.players_pieces_left
            dic_pieces_1 = dic_player_pieces[players_id[0]]
            dic_pieces_2 = dic_player_pieces[players_id[1]]
            pieces = ['RC', 'RR', 'GC', 'GR', 'BC', 'BR', 'YC', 'YR']
            nb_pieces_1, nb_pieces_2 = sum(dic_pieces_1[p] for p in pieces), sum(dic_pieces_2[p] for p in pieces)

            if nb_pieces_1 + nb_pieces_2 >= 12:
                depth = 4
            else:
                depth = 6
            
            _, best_action = self.alpha_beta_minimax(current_state, depth, float('-inf'), float('inf'), True)
            return best_action
    
    def count_threat(self, state: GameState, player):
        rep = state.get_rep()
        board = rep.get_env()
        d = rep.get_dimensions()
        threats = 0
        for i in range(d[0]):
            for j in range(d[1]):
                if state.in_board((i, j)) and (not (i, j) in board or board[i,j].get_owner_id() != player) and state.piece_type_match("C", (i, j)) and self.check_almost_divercite_city(state, (i, j), 2):
                    threats += 1
        return threats
    
    def count_opportunity(self, state: GameState, player):
        rep = state.get_rep()
        board = rep.get_env()
        d = rep.get_dimensions()
        opp = 0
        for i in range(d[0]):
            for j in range(d[1]):
                if state.in_board((i, j)) and (i, j) in board and board[i, j] and board[i,j].get_owner_id() == player and state.piece_type_match("C", (i, j)) and self.check_almost_divercite_city(state, (i, j), 3):
                    opp += 1
        return opp
    
    def count_diff(self, state: GameState, player):
        rep = state.get_rep()
        board = rep.get_env()
        d = rep.get_dimensions()
        threats = 0
        opp = 0
        for i in range(d[0]):
            for j in range(d[1]):
                if state.in_board((i, j)) and (not (i, j) in board or board[i,j].get_owner_id() != player) and state.piece_type_match("C", (i, j)) and self.check_almost_divercite_city(state, (i, j), 3):
                    threats += 1
                if state.in_board((i, j)) and (i, j) in board and board[i, j] and board[i,j].get_owner_id() == player and state.piece_type_match("C", (i, j)) and self.check_almost_divercite_city(state, (i, j), 3):
                    opp += 1
        return opp - threats

    def evaluate_state(self, state: GameState) -> float:
        """
        Evaluate the game state and return a heuristic value.
        """
        players = state.players
        players_id = [p.get_id() for p in players]
        player_id = self.get_id()

        player_score = state.scores[self.get_id()]
        opponent_score = state.scores[players_id[0]] if players_id[0] != player_id else state.scores[players_id[1]]
        
        dic_player_pieces = state.players_pieces_left
        dic_pieces_1 = dic_player_pieces[player_id]
        cite = ['RC', 'GC', 'BC', 'YC']
        ressource = ['RR', 'GR', 'BR', 'YR']
        nb_cite, nb_ressource = sum(dic_pieces_1[c] for c in cite), sum(dic_pieces_1[r] for r in ressource)
        
        nb_diff = 0
        if state.step > 10:
            nb_diff = self.count_diff(state, players_id)
        
        ressource_factor = dic_pieces_1["RR"] * dic_pieces_1["BC"] * dic_pieces_1["GR"] * dic_pieces_1["YR"]
        return (player_score - opponent_score) + 10*ressource_factor/81 + nb_cite * (1 - state.step/40) + nb_diff
