import random
import math
from seahorse.game.game_state import GameState

class TreeNode:
    def __init__(self, state: GameState, max_root_children=-1, parent=None):
        self.state = state
        self.parent = parent
        self.max_root_children = max_root_children
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        """ Check if all possible actions have been expanded. """
        actions = self.state.get_possible_light_actions()
        if self.parent is None and self.max_root_children > -1:
            return len(self.children) == min(len(actions), self.max_root_children)
        return len(self.children) == len(actions)

    def uct_value(self, exploration_constant=math.sqrt(2)):
        """ Calculate the UCT value for this node. """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
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
        if untried_actions:
            action = random.choice(untried_actions)
            next_state = self.state.apply_action(action)
            child_node = TreeNode(next_state, parent=self)
            self.children[action] = child_node
            return child_node
        return None

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
