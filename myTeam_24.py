# myTeam.py
# ---------
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
import math

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from distanceCalculator import manhattanDistance
from sympy import symbols, Eq, solve, re, im, I, var, solveset, N

import game
from util import nearestPoint
from baselineTeam import ReflexCaptureAgent

#################
# Team creation #
#################

gameStateOne = 0


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class SupremeAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.maxDepth = 5
        self.positions = []

        self.enemyInitialPositions = []

        self.boundary_top = True  # to change
        if gameState.getAgentState(self.index).getPosition()[0] == 1:  # to change
            self.isRed = True  # to change
        else:  # to change
            self.isRed = False  # to change

        self.boundaries = self.boundaryTravel(gameState)  # to change

        # Prediction of future states for enemies
        self.legalPositions = gameState.getWalls().asList(False)  # all possible legal positions for enemies
        self.obs = {}  # dictionary with observed values
        for enemy in self.getOpponents(gameState):  # add enemy initial positions and initialize
            self.enemyInitialPositions.append(gameState.getInitialAgentPosition(enemy))
            self.initBelief(enemy, gameState.getInitialAgentPosition(enemy))
        CaptureAgent.registerInitialState(self, gameState)

    # to change
    def boundaryTravel(self, gameState):  # to change
        """
        Returns two points that act as a boundary line along which the agent travels
        """
        walls = gameState.getWalls().asList()
        max_y = max([wall[1] for wall in walls])

        if not self.isRed:
            mid_x = (max([wall[0] for wall in walls]) / 2) + 2
        else:
            mid_x = max([wall[0] for wall in walls]) / 2

        walls = gameState.getWalls().asList()

        # lower bound is 1/3 of grid. Upper bound is 2/3 of grid
        lower = max_y / 3
        upper = (max_y * 2) / 3

        # If the positions are illegal states, add 1 to get a legal state
        while (mid_x, lower) in walls:
            lower += 1
        while (mid_x, upper) in walls:
            upper -= 1

        return (round(mid_x), round(lower)), (round(mid_x), round(upper))

    def observe(self, enemy, gameState):
        """
        Probabilities updated dependent upon distances between agents and distance observations
        """
        noise = gameState.getAgentDistances()  # noisy distances for all agents
        noisyDistance = noise[enemy]  # noisy distances for enemy agents
        myPos = gameState.getAgentPosition(self.index)  # position for our agent
        myTeamIndex = self.getTeam(gameState)
        myTeamPos = [gameState.getAgentPosition(team) for team in myTeamIndex]
        allObs = util.Counter()  # 'smart' dictionary with all observations
        for pos in self.legalPositions:  # for legal positions
            # if any friendly agent
            teamDist = [team for team in myTeamPos if util.manhattanDistance(team, pos) <= 5]
            if teamDist:  # we don't consider close points
                allObs[pos] = 0.0
            else:
                trueDistance = util.manhattanDistance(myPos, pos)  # true distance between agent and legal action
                posProb = gameState.getDistanceProb(trueDistance, noisyDistance)  # probability of pos given t/n dist
                allObs[pos] = posProb * self.obs[enemy][pos]
                # p(true distance|noisy distance) noisy dist = p(noisy distance|true distance) * true dist
        if allObs.totalCount():  # Note, totalcount() : Returns the sum of counts for all keys
            allObs.normalize()
            self.obs[enemy] = allObs
        else:
            self.initBelief(enemy, gameState.getInitialAgentPosition(enemy))

    def initBelief(self, enemy, startPos):
        """
        Set probability distributions for enemy positions
        """
        self.obs[enemy] = util.Counter()
        self.obs[enemy][startPos] = 1.0

    def setTruePosition(self, enemy, pos):
        """
        When true position is known, this set enemies true pos.
        """
        trueObs = util.Counter()
        trueObs[pos] = 1.0
        self.obs[enemy] = trueObs

    def getApproxPosition(self, enemy):
        """
        Get highest probable enemy position
        """
        probs = list(self.obs.items())  # list of all observed items
        if probs.count(max(probs)) < 5:
            return self.obs[enemy].argMax()
        else:
            return None

    def getPossiblePosition(self, x, y):
        """
        Return possible successor positions given x,y coordinates.
        (except stop, i.e. DIRECTIONS.Stop)
        """
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def timeStep(self, enemy, gameState):
        """
        Bayseian inference distribution of enemies positions for each timestep
        """

        allObs = util.Counter()  # 'smart' dictionary, to contain all posterior observations
        for prevPos, prevProb in self.obs[enemy].items():  # for posterior position and probabilities
            newObs = util.Counter()  # 'smart' dictionary with new observations

            for pos in self.getPossiblePosition(prevPos[0], prevPos[1]):  # for possible positions
                if pos in self.legalPositions:  # if said position is legal
                    newObs[pos] = 1.0  # e.g. newObs : {(1, 3): 1.0, (1, 1): 1.0}
            # note: normalize is needed in order to not get probabilities > 1 for some key in the dictionary
            newObs.normalize()  # Edits the counter such that the total count of all keys sums to 1.
            for newPos, newProb in newObs.items():  # for new positions and probabilities
                allObs[newPos] += newProb * prevProb  # inferred positions

        invaders = self.getInvaders(gameState)  # number of enemies on our side
        enemyState = gameState.getAgentState(enemy)  # state of enemies
        if enemyState.isPacman:  # if enemy is pacman, i.e. it eats foods
            takenFood = self.getFoodDiff(gameState)  # it eats this amount of food
            if takenFood:
                for food in takenFood:
                    allObs[food] = 1.0 / invaders
                allObs.normalize()

        self.obs[enemy] = allObs

    def getInvaders(self, gameState):
        """
        Return number of enemy agents on 'our' side
        - int
        """
        enemyIndex = gameState.getAgentState
        enemies = [enemyIndex(i) for i in self.getOpponents(gameState)]
        enemyOnOurSide = [x for x in enemies if x.isPacman]
        invaders = len(enemyOnOurSide)
        return invaders

    def getFoodDiff(self, gameState):
        foods = self.getFoodYouAreDefending(gameState).asList()  # foods we protect
        postObs = self.getPreviousObservation()  # GameState object corresponding to the last state this agent saw
        if postObs is not None:
            postFood = self.getFoodYouAreDefending(postObs).asList()
        else:
            postFood = list()
        takenFood = []
        if len(foods) < len(postFood):  # if food is eaten
            takenFood = list(set(postFood) - set(foods))  # update takenFood list
        return takenFood

    def getEnemyPosition(self, gameState):

        enemyPosition = []  # list with enemies
        enemyIndex = self.getOpponents(gameState)  # enemy index, i.e. 0,2 or 1,3 depending on team

        for enemy in enemyIndex:  # loop through opponents indices
            pos = gameState.getAgentPosition(enemy)  # set pos = real pos (if distance < 5 )
            if not pos:  # if we don't find the true position
                self.timeStep(enemy, gameState)
                self.observe(enemy, gameState)
            else:  # if we find the true position
                self.setTruePosition(enemy, pos)  # set true position ----

        for enemy in enemyIndex:
            anticipatedPos = self.getApproxPosition(enemy)
            enemyGameState = gameState.getAgentState(enemy) if anticipatedPos else None
            enemyPosition.append((enemyGameState, anticipatedPos))

        return enemyPosition

    def getEnemyAction(self, gameState):
        """
        Returns enemies possible actions
        prel: hardcoded agents, needs to change
        Double the code just because were calling two agents
        """

        directions = game.Actions._directions
        enemy = self.getEnemyPosition(gameState)
        enemyOne = enemy[0][1]
        enemyTwo = enemy[1][1]

        actions = []
        actionsPos = []
        for direction in directions:
            action = directions[direction]
            new_pos = (int(enemyOne[0] + action[0]),
                       int(enemyOne[1] + action[1]))
            if new_pos in self.legalPositions:
                actions.append(direction.title())
                actionsPos.append(new_pos)

        actions_ = []
        actionsPos_ = []
        for direction in directions:
            action = directions[direction]
            new_pos = (int(enemyTwo[0] + action[0]),
                       int(enemyTwo[1] + action[1]))
            if new_pos in self.legalPositions:
                actions_.append(direction.title())
                actionsPos_.append(new_pos)

        return actions, actionsPos, actions_, actionsPos_

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        # print("index:", self.index)
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getLegalActions(self, state, agentIndex):
        """
        Returns a list of legal actions (which are both possible & allowed)
        """
        agentState = state.getAgentState(agentIndex)
        agentState2 = state.getAgentState(2)

        conf = agentState.configuration
        possibleActions = self.getPossibleActions(conf, state.data.layout.walls)
        return self.filterForAllowedActions(agentState, possibleActions)

    def filterForAllowedActions(self, agentState, possibleActions):
        return possibleActions

    def getPossibleActions(self, config, walls):
        possible = []
        x, y = config.pos  ##########
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int) > .001):
            return [config.getDirection()]

        for dir, vec in game.Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        return possible

    def getMinDistanceToEnemies(self, gameState):
        """
        return distance between friendly agent and closest enemy
        """
        friendPos = gameState.getAgentPosition(self.index)  # position of friend index
        enemyPosition = self.getEnemyPosition(gameState)  # position and config(if possible) of both enemies
        enemyOnePos = enemyPosition[0][1]  # position of enemy one
        enemyTwoPos = enemyPosition[1][1]  # position of enemy two

        enemyOneDist = self.getMazeDistance(friendPos, enemyOnePos)
        enemyTwoDist = self.getMazeDistance(friendPos, enemyTwoPos)

        minDist = min(enemyOneDist, enemyTwoDist)
        return minDist

    def getDistanceToEnemies(self, gameState):
        """
        return distance between friendly agent and closest enemy
        """
        friendPos = gameState.getAgentPosition(self.index)  # position of friend index
        enemyPosition = self.getEnemyPosition(gameState)  # position and config(if possible) of both enemies
        enemyOnePos = enemyPosition[0][1]  # position of enemy one
        enemyTwoPos = enemyPosition[1][1]  # position of enemy two

        enemyOneDist = self.getMazeDistance(friendPos, enemyOnePos)
        enemyTwoDist = self.getMazeDistance(friendPos, enemyTwoPos)

        return enemyOneDist, enemyTwoDist

    def minimaxTest(self, gameState, friend=True, depth=0, parent=None, alpha=-float('inf'), beta=float('inf')):
        """
        minimax algorithm
        """
        currentBest = None

        if depth > 2:
            parentLegalActions = [action for action in parent.getLegalActions(self.index) if action != Directions.STOP]
            score = [self.evaluate(parent, x) for x in parentLegalActions]  # evaluate next possible steps
            maxValue = max(score)
            bestActions = [a for a, v in zip(parentLegalActions, score) if v == maxValue]  # act of the highest score
            return maxValue, bestActions
        elif friend:
            maxEval = -float('inf')
            friendLegalActions = [action for action in gameState.getLegalActions(self.index) if
                                  action != Directions.STOP]
            children = [self.getSuccessor(gameState, action) for action in friendLegalActions]
            parent = gameState
            for child in children:
                eval = self.minimaxTest(child, False, depth + 1, parent, alpha, beta)[0]
                maxEval = max(maxEval, eval)

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            parentLegalActions = [action for action in parent.getLegalActions(self.index) if action != Directions.STOP]
            score = [self.evaluate(parent, x) for x in parentLegalActions]  # evaluate next possible steps
            maxValue = max(score)
            bestActions = [a for a, v in zip(parentLegalActions, score) if v == maxValue]  # act of the highest score
            return maxEval, bestActions
        else:
            '''
            Could implement dependence on opponents player here, 
            e.g. minimize heuristic 
            alt. use other evaluation function for opponents since the imp.
            version use gameState
            '''
            minEval = float('inf')
            eval = self.minimaxTest(gameState, True, depth + 1, parent, alpha, beta)[0]
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                # not needed as of now
                pass

            # minEval = float('inf')
            return minEval, currentBest

    def minimax(self, gameState):
        """
        Function used for development of algorithm
        and comparing purposes.
        - Currently: returns minimax
        """
        # ---- indices ---- #
        friendIndex = self.getTeam(gameState)  # friend Index [1, 3]
        enemyIndex = self.getOpponents(gameState)  # enemy  Index [0, 2]

        # ---- positions ---- #
        friendPos = gameState.getAgentPosition(self.index)  # position of friend index
        enemyPosition = self.getEnemyPosition(gameState)  # position and config(if possible) of both enemies
        enemyOnePos = enemyPosition[0][1]  # position of enemy one
        enemyTwoPos = enemyPosition[1][1]  # position of enemy two

        # ---- actions ---- #
        friendLegalActions = [action for action in gameState.getLegalActions(self.index) if action != Directions.STOP]
        enemyLegalActionsList = self.getEnemyAction(gameState)  # list of actions and cons. pos of enemies
        enemyOneActions = enemyLegalActionsList[0]
        enemyTwoActions = enemyLegalActionsList[2]

        # ---- minimax ---- #
        minimaxEval = self.minimaxTest(gameState)

        '''
        loop through all possible next actions, evaluate the score of said actions
        and return the best action according to the score
         - Can be used to compare efficiency of minimax implementation 
         (since this is basically choose the next best action, whilst minimax 
         encompass evaluation of future moves)
        '''
        score = [self.evaluate(gameState, x) for x in friendLegalActions]  # evaluate next possible steps
        maxValue = max(score)
        bestActions = [a for a, v in zip(friendLegalActions, score) if v == maxValue]  # act of the highest score
        # print("maxValue Simple :", maxValue,       "maxValue simple : ", bestActions)
        # print("minimax values  :", minimaxEval[0], "minimax action  : ", minimaxEval[1])

        # ---- printing ---- #
        # print("friendPos index:", self.index, gameState.getAgentPosition(self.index))
        # print("enemyOnePos", enemyOnePos)
        # print("enemyTwoPos", enemyTwoPos)
        # print("friendLegalActions", friendLegalActions)
        # print("enemyLegalActionsList", enemyLegalActionsList)
        # print("enemyOneActions", enemyOneActions)
        # print("enemyTwoActions", enemyTwoActions)

        return random.choice(minimaxEval[1])


class DefensiveAgent(SupremeAgent):
    """
    Defensive Agent : Tries to defend from invaders
    """

    def chooseAction(self, gameState):
        start = time.time()
        nextAction = self.minimax(gameState)
        #print("defensive agent", time.time() - start)
        return nextAction

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        #print("featurs", features)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        enemyIndex = self.getOpponents(gameState)  # enemy index, i.e. 0,2 or 1,3 depending on team
        numInvaders = self.getInvaders(gameState)

        boundaries = self.boundaries  # to change

        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0  # if pacman, can't defend

        # to change below
        if numInvaders == 0:
            if self.boundary_top is True:
                bound = boundaries[0]
            else:
                bound = boundaries[1]
            if myPos == bound:
                self.boundary_top = not (self.boundary_top)
            features['bound'] = self.getMazeDistance(myPos, bound)

        # Computes distance to invaders we can see and their distance to the food we are defending
        if not myState.isPacman and myState.scaredTimer > 0:
            foodList = self.getFood(successor).asList()
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                minDistance = min([self.getMazeDistance(myPos, food)
                                   for food in foodList])
                features['distanceToFood'] = minDistance + 1
            features['defenseFoodDistance'] = 0.
        # to change above

        if numInvaders > 0:  # if we are invaded
            dist = self.getMinDistanceToEnemies(gameState)  # min distance to enemy
            if dist == 0:
                dist = -100
            features['invaderDistance'] = self.getMinDistanceToEnemies(gameState) + 1

        if self.getMinDistanceToEnemies(gameState) < 50:
            features['invaderDistance'] = self.getMinDistanceToEnemies(gameState) + 1


        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance
            features['foodRemaining'] = len(foodList)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 50, 'invaderDistance': -500,
                'distanceToFood': -1, 'defenseFoodDistance': -8, 'bound': -20,
                'stop': -100, 'reverse': -15}


class OffensiveAgent(SupremeAgent):
    """
    Offensive Agent : Tries to bring home food from the opponents side
    """

    def chooseAction(self, gameState):
        start = time.time()
        nextAction = self.minimax(gameState)
        print("offensive agent", time.time() - start)
        return nextAction

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights


    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes distance to enemy ghosts
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition()
                  != None]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]

        features['invaderDistance'] = 0.0
        if len(invaders) > 0:
            features['invaderDistance'] = min([self.getMazeDistance(
                myPos, invader.getPosition()) for invader in invaders]) + 1

        if len(ghosts) > 0:
            ghostEval = 0.0
            scaredDistance = 0.0
            regGhosts = [ghost for ghost in ghosts if ghost.scaredTimer == 0]
            scaredGhosts = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
            if len(regGhosts) > 0:
                ghostEval = min([self.getMazeDistance(
                    myPos, ghost.getPosition()) for ghost in regGhosts])
                if ghostEval <= 1:
                    ghostEval = -float('inf')

            if len(scaredGhosts) > 0:
                scaredDistance = min([self.getMazeDistance(
                    myPos, ghost.getPosition()) for ghost in scaredGhosts])
            if scaredDistance < ghostEval or ghostEval == 0:

                if scaredDistance == 0:
                    features['ghostScared'] = -10
            features['distanceToGhost'] = ghostEval

        # Compute distance to the nearest food
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance
            features['foodRemaining'] = len(foodList)

        # Compute distance to capsules
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            minDistance = min([self.getMazeDistance(myPos, capsule)
                               for capsule in capsules])
            if minDistance == 0:
                minDistance = -100
            features['distanceToCapsules'] = minDistance

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'invaderDistance': -50, 'distanceToFood': -1,
                'foodRemaining': -1, 'distanceToGhost': 2, 'ghostScared': -1,
                'distanceToCapsules': -1, 'stop': -100, 'reverse': -20}
