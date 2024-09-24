from __future__ import annotations
import json
from seahorse.game.action import Action
from seahorse.utils.serializer import Serializable
from player_divercite import PlayerDivercite
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses the Minimax algorithm with Alpha-Beta pruning.

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
        self.evaluation_cache = {}  # Cache for storing evaluations of game states

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Use the Minimax algorithm with Alpha-Beta pruning to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by Minimax with Alpha-Beta pruning.
        """

        def minimax(state: GameState, depth: int, maximizing_player: bool, alpha: float, beta: float) -> float:
            # Check for cached evaluation
            state_id = id(state)
            if depth == 0 or state.is_done():
                return self.evaluate_state(state)

            if maximizing_player:
                max_eval = float('-inf')
                for action in state.get_possible_light_actions():
                    next_state = state.apply_action(action)
                    eval = minimax(next_state, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta cut-off
                return max_eval
            else:
                min_eval = float('inf')
                for action in state.get_possible_light_actions():
                    next_state = state.apply_action(action)
                    eval = minimax(next_state, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha cut-off
                return min_eval

        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        possible_actions = list(current_state.get_possible_light_actions())
        possible_actions.sort(key=lambda action: self.evaluate_state(current_state.apply_action(action)), reverse=True)

        for action in possible_actions:
            next_state = current_state.apply_action(action)
            action_value = minimax(next_state, 3, False, alpha, beta)  ############ profondeur
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
        
        state_id = id(state)
        if state_id in self.evaluation_cache:
            return self.evaluation_cache[state_id]

        
        heuristic_value = state.scores[self.get_id()]  
        self.evaluation_cache[state_id] = heuristic_value  
        return heuristic_value
