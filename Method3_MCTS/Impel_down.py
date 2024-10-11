import random
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from player_divercite import PlayerDivercite

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses the Minimax algorithm with alpha-beta pruning and MCTS for the first 10 moves.
    """

    def __init__(self, piece_type: str, name: str = "AlphaBetaPlayer"):
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Compute action using MCTS for the first 10 moves, then alpha-beta pruning.
        """
        def mcts(state: GameState, simulations: int = 1000) -> Action:      ### Attention que 1000 simulations peut-être pas assez
            """ Perform MCTS to determine the best action. """
            action_counts = {action: 0 for action in state.get_possible_light_actions()}
            action_values = {action: 0 for action in state.get_possible_light_actions()}

            for _ in range(simulations):
                # Convert possible actions to a list
                possible_actions_list = list(action_counts.keys())
                action = random.choice(possible_actions_list)
                next_state = state.apply_action(action)

                # Simulate the game to completion from the next state
                while not next_state.is_done():
                    possible_actions = next_state.get_possible_light_actions()
                    # Convert possible actions to a list
                    possible_actions_list = list(possible_actions)
                    random_action = random.choice(possible_actions_list)
                    next_state = next_state.apply_action(random_action)

                # Use the evaluation function to determine the outcome of the simulation
                outcome = self.evaluate_state(next_state)
                action_counts[action] += 1
                action_values[action] += outcome

            # Calculate average values and choose the best action
            best_action = max(action_values, key=lambda a: action_values[a] / action_counts[a])
            return best_action


        def alpha_beta_minimax(state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
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
                    eval, _ = alpha_beta_minimax(next_state, depth - 1, alpha, beta, False)
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
                    eval, _ = alpha_beta_minimax(next_state, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_action = action
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Coupure
                return min_eval, best_action  # Return value and best action

        # Utiliser MCTS pour les 10 premiers coups
        if current_state.get_step() < 10:
            return mcts(current_state)

        # Pour les coups suivants, utiliser alpha-beta
        else:
            players = current_state.players
            players_id = [p.get_id() for p in players]
            dic_player_pieces = current_state.players_pieces_left
            dic_pieces_1 = dic_player_pieces[players_id[0]]
            dic_pieces_2 = dic_player_pieces[players_id[1]]
            pieces = ['RC', 'RR', 'GC', 'GR', 'BC', 'BR', 'YC', 'YR']
            nb_pieces_1, nb_pieces_2 = sum(dic_pieces_1[p] for p in pieces), sum(dic_pieces_2[p] for p in pieces)

            # Ajuster la profondeur en fonction du nombre de pièces restantes
            if nb_pieces_1 + nb_pieces_2 >= 12:
                depth = 4
            else:
                depth = 6
            
            _, best_action = alpha_beta_minimax(current_state, depth, float('-inf'), float('inf'), True)
            return best_action

    def evaluate_state(self, state: GameState) -> float:
        """
        Evaluate the game state and return a heuristic value.
        """
        players = state.players
        players_id = [p.get_id() for p in players]
        player_id = self.get_id()

        player_score = state.scores[self.get_id()]
        opponent_score = state.scores[players_id[0]] if players_id[0] != player_id else state.scores[players_id[1]]
        
        players = state.players
        dic_player_pieces = state.players_pieces_left
        dic_pieces_1 = dic_player_pieces[player_id]
        cite = ['RC', 'GC', 'BC', 'YC']
        ressource = ['RR', 'GR', 'BR', 'YR']
        nb_cite, nb_ressource = sum(dic_pieces_1[c] for c in cite), sum(dic_pieces_1[r] for r in ressource)
        
        return player_score - opponent_score + (1 - 4 * state.step / 40) * nb_cite + (1 + 4 * state.step / 40) * nb_ressource
