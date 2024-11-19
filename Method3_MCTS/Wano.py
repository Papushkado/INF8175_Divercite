import random
import time
from functools import lru_cache
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from player_divercite import PlayerDivercite


class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "AlphaBetaOptimized"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameState, max_time: float = 90, **kwargs) -> Action:
        if current_state.get_step() < 2:
            possible_actions = current_state.get_possible_light_actions()
            city_actions = [action for action in possible_actions if action.data["piece"] in ['RC', 'GC', 'BC', 'YC']]
            return random.choice(city_actions)

        return self.iterative_deepening(current_state, max_time)

    def iterative_deepening(self, state: GameState, max_time: float) -> Action:
        """
        Utilise l'itération progressive pour déterminer le meilleur coup avec une limite de temps.
        """
        start_time = time.time()
        step = state.get_step()

        # Déterminer la profondeur initiale
        if step < 7:
            depth = 3
        elif step < 12:
            depth = 4
        else:
            depth = 6

        max_depth = 15  # Limite maximale de profondeur pour éviter les calculs excessifs
        best_action = None

        while time.time() - start_time < max_time:
            if depth > max_depth:
                break  # Empêcher une boucle infinie en limitant la profondeur
            try:
                _, current_best_action = self.alpha_beta_minimax(
                    state, depth, float('-inf'), float('inf'), True
                )
                if current_best_action is not None:
                    best_action = current_best_action
            except Exception as e:
                print(f"Erreur lors de la recherche à profondeur {depth}: {e}")
                break  # Sortir de la boucle en cas de problème imprévu

            depth += 1  # Augmenter la profondeur pour la prochaine itération

        # Si aucune action n'a été trouvée, choisir une action aléatoire
        if best_action is None:
            possible_actions = state.get_possible_light_actions()
            return random.choice(possible_actions)

        return best_action


    def alpha_beta_minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing: bool):
        if depth == 0 or state.is_done():
            return self.evaluate_state(state), None

        actions = state.get_possible_light_actions()

        if len(actions) > 5:
            actions = sorted(
                actions, key=lambda a: self.evaluate_state(state.apply_action(a)), reverse=maximizing
            )

        best_action = None
        if maximizing:
            max_eval = float('-inf')
            for action in actions:
                next_state = state.apply_action(action)
                eval, _ = self.alpha_beta_minimax(next_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in actions:
                next_state = state.apply_action(action)
                eval, _ = self.alpha_beta_minimax(next_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    @lru_cache(maxsize=1000)
    def evaluate_state(self, state: GameState) -> float:
        players = state.players
        player_id = self.get_id()
        player_score = state.scores[player_id]
        opponent_score = sum(score for pid, score in state.scores.items() if pid != player_id)

        dic_pieces = state.players_pieces_left[player_id]
        nb_cite = sum(dic_pieces[c] for c in ['RC', 'GC', 'BC', 'YC'])
        nb_ressource = sum(dic_pieces[r] for r in ['RR', 'GR', 'BR', 'YR'])

        return (
            player_score - opponent_score
            + (1 - 4 * state.step / 40) * nb_cite
            + (1 + 4 * state.step / 40) * nb_ressource
        )
