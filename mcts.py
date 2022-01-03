from __future__ import division

import time
import math
import random
import numpy as np
from reversi_state import Action

# We need a policy so we play until the game end.
def randomPolicy(state):
    # Until we have a winner
    while not state.isTerminal():
        try:
            # Execute a random action
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    # Return the reward
    return state.getReward()

# The same as above but this time we use a pre-trained policy, not a random one.
def modelPolicy(model):
    def sampleModel(state):
        while not state.isTerminal():
            try:
                action_probs = model.eval()(state.board.reshape(1, 1, *state.board.shape)).detach().numpy()[0] * state.get_actions_mask()
                action_probs = action_probs/action_probs.sum()
                action = np.random.choice(len(action_probs), p=action_probs)
                coded_action = [action % state.board.shape[0], action // state.board.shape[0]]
                action = Action(coded_action , state.getCurrentPlayer())
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            state = state.takeAction(action)
        return state.getReward()
    return sampleModel

# Create a node for a given state
class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        # How many times we visited the node
        self.numVisits = 0
        self.totalReward = 0
        # For a given state, we need each action with it next-state
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    # With this function we run the algorithm.
    def search(self, initialState, needDetails=False):
        # It first look for the first state of the game
        self.root = treeNode(initialState, None)
        # And then start executing iterations 
        # For a limit time
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            # Or for a limit number of iterations
            for i in range(self.searchLimit):
                self.executeRound()

        
        if needDetails:
            node_actions, bestChild = self.getBestChild(self.root, 0, verbose=needDetails)
            #return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
            action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
            return node_actions, action
        else:
            bestChild = self.getBestChild(self.root, 0, verbose=needDetails)
            action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        # Select a  node
        node = self.selectNode(self.root)

        # play from that state and save the reward
        reward = self.rollout(node.state)

        # backprogate the reward for all the state where it played
        self.backpropogate(node, reward)


    def selectNode(self, node):
        # Until you find one of the last state, or if the state is not fully expanded, continue selecting new states
        while not node.isTerminal:
            if node.isFullyExpanded:
                # Select the best action for that state
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        # So if the state is not fully expanded, we need to look for all actions for that given state
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                # generate a new node for the given action
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            # Add one visit to the node
            node.numVisits += 1
            # Add the reward
            node.totalReward += reward
            # Change to the previous state
            node = node.parent

    def getBestChild(self, node, explorationValue, verbose=0):
        # Here we make the calculation to know which is the best child (best action)
        bestValue = float("-inf")
        bestNodes = []
        node_actions = {}

        for key, child in node.children.items():
            # Calculate the value of that state
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if verbose:
                node_actions[(key.action[0], key.action[1], key.player)] = (nodeValue, child.totalReward, child.numVisits, (child.totalReward + child.numVisits)/2/child.numVisits)
            if nodeValue > bestValue:
                # If the value calculated is greater than the previous one:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                # If there is a draw between child values
                bestNodes.append(child)
        if verbose:
            return node_actions, random.choice(bestNodes)
        return random.choice(bestNodes)