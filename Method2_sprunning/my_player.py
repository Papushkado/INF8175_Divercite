from __future__ import annotations
import json
import random
from seahorse.game.action import Action
from seahorse.utils.serializer import Serializable
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError


class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses the Minimax algorithm with alpha-beta pruning.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "AlphaBetaPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "AlphaBetaPlayer")
        """
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Use the minimax algorithm with alpha-beta pruning to choose the best action.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by the alpha-beta algorithm.
        """

        def alpha_beta_minimax(state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
            if depth == 0 or state.is_done():
                return self.evaluate_state(state)

            if maximizing_player:
                max_eval = float('-inf')
                for action in state.get_possible_light_actions():
                    next_state = state.apply_action(action)
                    eval = alpha_beta_minimax(next_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  
                return max_eval
            else:
                min_eval = float('inf')
                for action in state.get_possible_light_actions():
                    next_state = state.apply_action(action)
                    eval = alpha_beta_minimax(next_state, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  
                return min_eval

        best_action = None
        best_value = float('-inf')

        # Random action si on joue premier (attention on peut y réflechir, parce que dans un coin c'est peut-être plus worth)
        if current_state.get_step() < 2 :
            possible_actions = current_state.get_possible_light_actions()
            return random.choice(list(possible_actions))
        
        else:
            for action in current_state.get_possible_light_actions():
                next_state = current_state.apply_action(action)

                # Check number of remaining pieces to adjust depth dynamically
                players = current_state.players
                players_id = [p.get_id() for p in players]
                dic_player_pieces = current_state.players_pieces_left
                dic_pieces_1 = dic_player_pieces[players_id[0]]
                dic_pieces_2 = dic_player_pieces[players_id[1]]
                pieces = ['RC', 'RR', 'GC', 'GR', 'BC', 'BR', 'YC', 'YR']
                nb_pieces_1, nb_pieces_2 = 0, 0

                for p in pieces:
                    nb_pieces_1 += dic_pieces_1[p]
                    nb_pieces_2 += dic_pieces_2[p]

                # Ajustement de la profondeur selon le facteur de branchement 
                if nb_pieces_1 + nb_pieces_2 >= 20:
                    action_value = alpha_beta_minimax(next_state, 3, float('-inf'), float('inf'), True)
                elif nb_pieces_1 + nb_pieces_2 >= 15:
                    action_value = alpha_beta_minimax(next_state, 4, float('-inf'), float('inf'), True)
                else:
                    action_value = alpha_beta_minimax(next_state, 5, float('-inf'), float('inf'), True)

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

        return best_action

    def evaluate_state(self, state: GameState) -> float:
        """
        Evaluate the game state and return a heuristic value.

        Args:
            state (GameState): The current game state.

        Returns:
            float: Heuristic value of the game state.
        """
        players = state.players  # Get both players
        players_id = [p.get_id() for p in players]
        player_id = self.get_id()

        player_score = state.scores[self.get_id()]
        opponent_score = state.scores[players_id[0]] if players_id[0] != player_id else state.scores[players_id[1]]

        return player_score - opponent_score
    # return state.scores[self.get_id()]   # Vraiment pas folle parce qu'on peut augmenter le score de l'adversaire

