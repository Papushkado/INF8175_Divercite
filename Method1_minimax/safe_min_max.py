from __future__ import annotations
import json
from seahorse.game.action import Action
from seahorse.utils.serializer import Serializable
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError


class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses the Minimax algorithm.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MinimaxPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "MinimaxPlayer")
        """
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        def minimax(state: GameState, depth: int, maximizing_player: bool) -> float:
            if depth == 0 or state.is_done():
                return self.evaluate_state(state)

            if maximizing_player:
                max_eval = float('-inf')
                for action in state.get_possible_light_actions():
                    next_state = state.apply_action(action)
                    eval = minimax(next_state, depth - 1, False)
                    max_eval = max(max_eval, eval)
                return max_eval
            else:
                min_eval = float('inf')
                for action in state.get_possible_light_actions():
                    next_state = state.apply_action(action)
                    eval = minimax(next_state, depth - 1, True)
                    min_eval = min(min_eval, eval)
                return min_eval

        best_action = None
        best_value = float('-inf')

        for action in current_state.get_possible_light_actions():
            next_state = current_state.apply_action(action)
            action_value = minimax(next_state, 2, True)  ################### Ici pour changer la profondeur et mettre à True car on veut maximiser
            if action_value > best_value:                # Pronfondeur 3 c'est trop, c'est tropico
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
        players = state.players # Dedans il y a le player adverse et le mien
        players_id = [p.get_id() for p in players]
        player_id = self.get_id()
        
        player_score = state.scores[self.get_id()]
        if players_id[0] == player_id:
            opponent_score = state.scores[players_id[1]]
        else : 
            opponent_score = state.scores[players_id[0]]
        
        
        return player_score - opponent_score  
    # return state.scores[self.get_id()]   # Vraiment pas folle parce qu'on peut augmenter le score de l'adversaire

