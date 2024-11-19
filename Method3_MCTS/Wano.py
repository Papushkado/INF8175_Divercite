import random
import time
from functools import lru_cache
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from player_divercite import PlayerDivercite


class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses the Minimax algorithm with alpha-beta pruning.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "AlphaBetaPlayer"):
        super().__init__(piece_type, name)
        self.transposition_table = {}  # Table de transposition pour mémoriser les états explorés

    def compute_action(self, current_state: GameState, max_time: float = 1.5, **kwargs) -> Action:
        """
        Use iterative deepening with alpha-beta pruning to choose the best action.
        """
        if current_state.get_step() < 2:
            # Pré-choisir une action parmi les cités si le jeu vient de commencer
            possible_actions = current_state.get_possible_light_actions()
            city_actions = [action for action in possible_actions if action.data["piece"] in ['RC', 'GC', 'BC', 'YC']]
            return random.choice(city_actions)

        # Sinon, utilise l'itération progressive pour déterminer le meilleur coup
        return self.iterative_deepening(current_state, max_time)

    def iterative_deepening(self, state: GameState, max_time: float) -> Action:
        """
        Utilise l'itération progressive pour déterminer le meilleur coup avec une limite de temps.
        """
        start_time = time.time()
        depth = 1
        best_action = None

        while time.time() - start_time < max_time:
            _, best_action = self.alpha_beta_minimax(state, depth, float('-inf'), float('inf'), True, start_time, max_time)
            depth += 1
        return best_action

    def alpha_beta_minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool, start_time: float, max_time: float):
        """
        Minimax algorithm with alpha-beta pruning, using a transposition table and action sorting.
        """
        state_key = self.generate_state_key(state)  # Génère une clé unique pour l'état
        if (state_key, depth) in self.transposition_table:
            return self.transposition_table[(state_key, depth)]

        # Arrêt si la limite de temps est dépassée
        if time.time() - start_time >= max_time:
            return float('-inf') if maximizing_player else float('inf'), None

        if depth == 0 or state.is_done():
            eval = self.evaluate_state(state)
            self.transposition_table[(state_key, depth)] = eval, None
            return eval, None

        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            actions = state.get_possible_light_actions()

            # Trie des actions pour explorer d'abord les plus prometteuses
            actions = sorted(actions, key=lambda a: self.evaluate_state(state.apply_action(a)), reverse=True)

            for action in actions:
                next_state = state.apply_action(action)
                eval, _ = self.alpha_beta_minimax(next_state, depth - 1, alpha, beta, False, start_time, max_time)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Coupure
            self.transposition_table[(state_key, depth)] = max_eval, best_action
            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = None
            actions = state.get_possible_light_actions()

            # Trie des actions pour explorer d'abord les plus prometteuses
            actions = sorted(actions, key=lambda a: self.evaluate_state(state.apply_action(a)))

            for action in actions:
                next_state = state.apply_action(action)
                eval, _ = self.alpha_beta_minimax(next_state, depth - 1, alpha, beta, True, start_time, max_time)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Coupure
            self.transposition_table[(state_key, depth)] = min_eval, best_action
            return min_eval, best_action

    @lru_cache(maxsize=None)
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

        # Pondération dynamique des cités et ressources en fonction de l'étape du jeu
        return player_score - opponent_score + (1 - 4 * state.step / 40) * nb_cite + (1 + 4 * state.step / 40) * nb_ressource

    def generate_state_key(self, state: GameState) -> tuple:
        """
        Generate a unique key for the game state based on critical components.
        """
        # Utilisation des attributs essentiels pour construire une clé unique
        board_repr = tuple(sorted((pos, piece.get_type()) for pos, piece in state.get_rep().get_env().items()))
        player_pieces = tuple(
            (player_id, tuple(sorted(pieces.items()))) for player_id, pieces in sorted(state.players_pieces_left.items())
        )
        scores = tuple(state.scores.values())
        return (state.step, state.next_player.get_id(), scores, board_repr, player_pieces)
