import random
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from player_divercite import PlayerDivercite
import MCTS_Steph

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses a hybrid strategy:
    MCTS for the first 10 moves, followed by alpha-beta pruning.
    """

    def __init__(self, piece_type: str, name: str = "AlphaBetaPlayer"):
        super().__init__(piece_type, name)

# Je teste avec des valeurs plus faibles pour que ça baisse
    def mcts(self, state: GameState, max_root_children=10, simulations=5000) -> Action:
        """ Perform MCTS using an improved implementation. """
        root = MCTS_Steph.TreeNode(state, max_root_children=max_root_children)
        
        for sim in range(simulations):
            print(f"\rMCTS Iteration: {sim + 1}/{simulations}", end='', flush=True)

            # Selection
            node = root.select()

            # Expansion
            if not node.state.is_done() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            outcome = self.evaluate_simulation(node.state)

            # Backpropagation
            while node:
                node.update(outcome)
                node = node.parent

        print("\n")
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

    def evaluate_simulation(self, state: GameState) -> float:
        """ Perform a heuristic-guided simulation from a given state. """
        current_state = state
        while not current_state.is_done():
            possible_actions = list(current_state.get_possible_light_actions())
            action_scores = [(a, self.evaluate_state(current_state.apply_action(a))) for a in possible_actions]

            # Weighted random selection based on heuristic evaluation
            total_score = sum(score for _, score in action_scores)
            if total_score > 0:
                probabilities = [score / total_score for _, score in action_scores]
                action = random.choices([a for a, _ in action_scores], weights=probabilities, k=1)[0]
            else:
                action = random.choice(possible_actions)

            current_state = current_state.apply_action(action)

        return self.evaluate_state(current_state)

    def alpha_beta_minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool):
        """ Optimized alpha-beta pruning with action ordering. """
        if depth == 0 or state.is_done():
            return self.evaluate_state(state), None

        best_action = None
        actions = list(state.get_possible_light_actions())

        # Sort actions based on heuristic evaluation
        actions.sort(key=lambda a: self.evaluate_state(state.apply_action(a)), reverse=maximizing_player)

        if maximizing_player:
            max_eval = float('-inf')
            for action in actions:
                next_state = state.apply_action(action)
                eval, _ = self.alpha_beta_minimax(next_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Pruning
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
                    break  # Pruning
            return min_eval, best_action

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Compute action using MCTS for the first 10 moves, then alpha-beta pruning.
        
        To avoid calculate to much at the begining of the game, we used the two first moves to uses our cities that we consider weaker than ressources
        Due to the geometry of the problem, we didn't find a good first moove (better than randomness) 
        """
        
        if current_state.get_step() < 2:
            possible_actions = current_state.get_possible_light_actions()
            # Filtrer pour ne garder que les actions qui placent une cité (RC, GC, BC, YC)
            city_actions = [action for action in possible_actions if action.data["piece"] in ['RC', 'GC', 'BC', 'YC']]
            return random.choice(city_actions)
### Modifier pour voir à partir de combien c'est bon
        if current_state.get_step() < 6:
            return self.mcts(current_state)

        if current_state.get_step() > 28: 
            depth = 4
        else: 
            depth = 6
        _, best_action = self.alpha_beta_minimax(current_state, depth, float('-inf'), float('inf'), True)
        return best_action

    def evaluate_state(self, state: GameState) -> float:
        """
        Evaluate the game state and return a heuristic value.
        """
        players = state.players
        player_id = self.get_id()
        opponent_id = [p.get_id() for p in players if p.get_id() != player_id][0]

        player_score = state.scores[player_id]
        opponent_score = state.scores[opponent_id]
        dic_pieces = state.players_pieces_left[player_id]

        cite = ['RC', 'GC', 'BC', 'YC']
        ressource = ['RR', 'GR', 'BR', 'YR']
        nb_cite = sum(dic_pieces[c] for c in cite)
        nb_ressource = sum(dic_pieces[r] for r in ressource)

        step_factor = 4 * state.step / 40
        return player_score - opponent_score + (1 - step_factor) * nb_cite + (1 + step_factor) * nb_ressource
