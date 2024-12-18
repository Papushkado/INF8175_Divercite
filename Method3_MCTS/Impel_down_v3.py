import random
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from player_divercite import PlayerDivercite
import math

class TreeNode:
    def __init__(self, state : GameState, max_root_children = -1, parent=None):
        self.state = state
        self.parent = parent
        self.max_root_children = max_root_children 
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        """ Check if all possible actions have been expanded. """
        if (self.parent == None and self.max_root_children > -1) :
            return len(self.children) == self.max_root_children
        else: 
            return len(self.children) == len(self.state.get_possible_light_actions())

    def uct_value(self, exploration_constant=math.sqrt(2)):
        """ Calculate the UCT value for this node. """
        if self.visits == 0:
            return float('inf')  # Ensure unvisited nodes are prioritized
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_constant=math.sqrt(2)):
        """ Select the child with the highest UCT value. """
        return max(self.children.values(), key=lambda child: child.uct_value(exploration_constant))

    def expand(self):
        """ Expand by adding a child for an untried action. """
        actions = self.state.get_possible_light_actions()
        untried_actions = [a for a in actions if a not in self.children]
        action = random.choice(untried_actions)
        next_state = self.state.apply_action(action)
        child_node = TreeNode(next_state, parent=self)
        self.children[action] = child_node
        return child_node

    def update(self, outcome):
        """ Update node statistics on backpropagation. """
        self.visits += 1
        self.value += outcome

    def select(self):
        """ Traverse the tree using UCT until reaching a leaf node. """
        node = self
        while not node.isLeaf() and node.is_fully_expanded():
            node = node.best_child()
        return node

    def isLeaf(self):
        """ Check if this node is a leaf (has no children). """
        return len(self.children) == 0
    
class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that uses the Minimax algorithm with alpha-beta pruning and MCTS for the first 10 moves.
    """

    def __init__(self, piece_type: str, name: str = "AlphaBetaPlayer"):
        super().__init__(piece_type, name)

    def mcts(self, state: GameState, simulations: int = 1000) -> Action:      ### Attention que 1000 simulations peut-être pas assez
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
    
    def simpleSimulation(self, node):
        current_state = node.state
        while not current_state.is_done():
            possible_actions = list(current_state.get_possible_light_actions())
            action = random.choice(possible_actions)
            current_state = current_state.apply_action(action)
        return self.evaluate_state(current_state)

    def heuristicsSimulation(self, node):
        current_state = node.state
        while not current_state.is_done():
            possible_actions = list(current_state.get_possible_light_actions())
        
            # Evaluate each possible next state
            action_scores = []
            for action in possible_actions:
                next_state = current_state.apply_action(action)
                score = self.evaluate_state(next_state)
                action_scores.append((action, score))
    
            # Calculate the total score for normalization
            total_score = sum(score for _, score in action_scores)
    
            if total_score > 0:
            # Weighted random choice based on normalized probabilities
                probabilities = [score / total_score for _, score in action_scores]
                action = random.choices([a for a, _ in action_scores], weights=probabilities, k=1)[0]
            else:
                # Fallback to uniform random choice if all scores are zero
                action = random.choice(possible_actions)
    
            # Apply the chosen action
            current_state = current_state.apply_action(action)

        return self.evaluate_state(current_state)

    def mcts_taylorsVersion(self, state : GameState, simple, max_root_children = -1, simulation = 1000):
        treePaine = TreeNode(state, max_root_children)
        if treePaine.parent == None and max_root_children > 0:
            actions = state.get_possible_light_actions()
            actions = sorted(actions, key=lambda a: self.evaluate_state(state.apply_action(a)), reverse=True)[:max_root_children]
            treePaine.children = {action: TreeNode(state.apply_action(action), parent=treePaine) for action in actions}
        for _ in range(simulation):
            print(f"\rMCTS Iteration: {_ + 1}/{simulation}, root children: {len(treePaine.children)}", end='', flush=True)
            if _ == simulation - 1:
                print("\n")
            #Select 
            node = treePaine.select()

            # 2. Expansion
            if not node.state.is_done() and not node.is_fully_expanded():
                node = node.expand()

            outcome = 0
            if simple:
                outcome = self.simpleSimulation(node)
            else:
                outcome = self.heuristicsSimulation(node)

            # Backpropagate
            while node:
                node.update(outcome)
                node = node.parent

        # Choose the action leading to the best child
        best_action = max(treePaine.children.items(), key=lambda item: item[1].visits)[0]
        return best_action
    
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
        Compute action using MCTS for the first 10 moves, then alpha-beta pruning.
        """

        if current_state.get_step() < 2:
            return self.greedy(current_state)
        # Utiliser MCTS pour les 10 premiers coups
        if current_state.get_step() < 17:
### Attention j'ai modifié ta version ici 
            #return self.mcts_taylorsVersion(current_state, True, 10, 20000)
            return self.mcts_taylorsVersion(current_state, True, 10, 5000)
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
            if nb_pieces_1 + nb_pieces_2 >= 26:
                depth = 4
            elif nb_pieces_1 + nb_pieces_2 >= 10:
                depth = 6
            else:
                depth = 7
            
            _, best_action = self.alpha_beta_minimax(current_state, depth, float('-inf'), float('inf'), True)
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
        
        dic_player_pieces = state.players_pieces_left
        dic_pieces_1 = dic_player_pieces[player_id]
        cite = ['RC', 'GC', 'BC', 'YC']
        ressource = ['RR', 'GR', 'BR', 'YR']
        nb_cite, nb_ressource = sum(dic_pieces_1[c] for c in cite), sum(dic_pieces_1[r] for r in ressource)
        
        return player_score - opponent_score + (1 - 4 * state.step / 40) * nb_cite + (1 + 4 * state.step / 40) * nb_ressource
            