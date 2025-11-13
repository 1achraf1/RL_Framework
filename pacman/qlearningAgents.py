# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld

import random, util, math
import numpy as np
import copy
from collections import defaultdict


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # Use defaultdict to default unseen state-action pairs to 0.0
        self.qValues = defaultdict(float)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          0.0 for unseen state-action pairs
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action) over legal actions.
          Return 0.0 if no legal actions (terminal state)
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        # max Q-value among all legal actions
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
          Break ties randomly. Return None if no legal actions.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Compute Q-values for all legal actions
        qValues = [(action, self.getQValue(state, action)) for action in legalActions]
        maxValue = max(q for _, q in qValues)

        # Gather all actions that have the max Q-value
        bestActions = [action for action, q in qValues if q == maxValue]

        # Break ties randomly
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Compute action using epsilon-greedy strategy
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # With probability epsilon, take a random action
        if flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
          Q-learning update rule:
          Q(s,a) := (1 - alpha) * Q(s,a) + alpha * [reward + discount * max_a' Q(s',a')]
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        qValue = 0.0
        for feature, value in features.items():
            qValue += self.weights[feature] * value
        return qValue

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition

           The update rule for approximate Q-learning is:
           difference = (reward + discount * max_a' Q(s',a')) - Q(s,a)
           w_i := w_i + alpha * difference * f_i(s,a)

           where f_i(s,a) is the value of feature i for state-action pair (s,a)
        """
        # Get features for current state-action pair
        features = self.featExtractor.getFeatures(state, action)

        # Calculate the temporal difference (TD error)
        currentQValue = self.getQValue(state, action)
        maxNextQValue = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount * maxNextQValue) - currentQValue

        # Update weights for each feature
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # Print weights for debugging
            print("Training completed. Final weights:")
            for feature, weight in self.weights.items():
                print(f"  {feature}: {weight:.4f}")