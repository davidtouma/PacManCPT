# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from turtle import clear
from capture import SONAR_NOISE_VALUES
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import AgentState, Directions
import game
from util import nearestPoint

#################
# Team creation #
#################
ANTICIPATER = []

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MyOffensiveAgent', second = 'MyDefensiveAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class MyCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    self.enemyStartPositions = []
    self.legalPositions = gameState.getWalls().asList(False)
    self.obs = {}
    for enemy in self.getOpponents(gameState):
      self.enemyStartPositions.append(gameState.getInitialAgentPosition(enemy))
      self.initalize(enemy, gameState.getInitialAgentPosition(enemy))
    

  def chooseAction(self, gameState):
    global ANTICIPATER
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())
    
    if self.index == 0 or self.index == 1 or len(ANTICIPATER) == 0:
          ANTICIPATER = self.getAnticipatedGhosts(gameState)

    pos_list = [ghostPos for ghost, ghostPos in ANTICIPATER]
    print(f"Ghosts Anticipated: {[ghostPos for ghost, ghostPos in ANTICIPATER]}")
    CaptureAgent.debugDraw(self, pos_list[0], (1,0,0), True)

    self.numCarrying = gameState.data.agentStates[self.index].numCarrying
    '''
    agentDist = gameState.getAgentDistances()
    enemyDist = agentDist[1::2]
    agentPos = gameState.getAgentPosition(0)
    #print(SONAR_NOISE_VALUES)
    #print(enemyDist, self.index)
    
    print("-------------")
    '''
   
    
    if self.numCarrying >= 5:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
    return random.choice(bestActions)
  
  
  ## Bayesian Inference Functions Starts   ###### THIS IMPLEMENTATION IS FROM https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py
  def initalize(self, enemy, startPos):
    """
    Uniformly initialize belief distributions for opponent positions.
    """
    self.obs[enemy] = util.Counter()
    self.obs[enemy][startPos] = 1.0

  def setTruePos(self, enemy, pos):
    """
    Fix the position of an opponent in an agent's belief distributions.
    """
    trueObs = util.Counter()
    trueObs[pos] = 1.0
    self.obs[enemy] = trueObs

  def elapseTime(self, enemy, gameState):
    """
    Elapse belief distributions for an agent's position by one time step.
    Assume opponents move randomly, but also check for any food lost from
    the previous turn.
    """
    possiblePos = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    allObs = util.Counter()
    for prevPos, prevProb in self.obs[enemy].items():
      newObs = util.Counter()
      for pos in possiblePos(prevPos[0], prevPos[1]):
        if pos in self.legalPositions:
          newObs[pos] = 1.0
      newObs.normalize()
      for newPos, newProb in newObs.items():
        allObs[newPos] += newProb * prevProb

    invaders = self.numberOfInvaders(gameState)
    enemyState = gameState.getAgentState(enemy)
    if enemyState.isPacman:
      eatenFood = self.getFoodDiff(gameState)
      if eatenFood:
        for food in eatenFood:
          allObs[food] = 1.0 / invaders
        allObs.normalize()

    self.obs[enemy] = allObs

  def getFoodDiff(self, gameState):
    foods = self.getFoodYouAreDefending(gameState).asList()
    prevFoods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList() \
      if self.getPreviousObservation() is not None else list()
    eatenFood = []
    if len(foods) < len(prevFoods):
      eatenFood = list(set(prevFoods) - set(foods))
    return eatenFood

  def observe(self, enemy, gameState):
    """
    Updates beliefs based on the distance observation and Pacman's
    position.
    """
    allnoise = gameState.getAgentDistances()
    noisyDistance = allnoise[enemy]
    myPos = gameState.getAgentPosition(self.index)
    teamPos = [gameState.getAgentPosition(team) for team in self.getTeam(gameState)]
    allObs = util.Counter()

    for pos in self.legalPositions:
      teamDist = [team for team in teamPos if util.manhattanDistance(team, pos) <= 5]
      if teamDist:
        allObs[pos] = 0.0
      else:
        trueDistance = util.manhattanDistance(myPos, pos)
        posProb = gameState.getDistanceProb(trueDistance, noisyDistance)
        allObs[pos] = posProb * self.obs[enemy][pos]

    if allObs.totalCount():
      allObs.normalize()
      self.obs[enemy] = allObs
    else:
      self.initalize(enemy, gameState.getInitialAgentPosition(enemy))

  def approxPos(self, enemy):
    """
    Return the highest probably  enemy position
    """
    values = list(self.obs.items())
    if values.count(max(values)) < 5:
      return self.obs[enemy].argMax()
    else:
      return None

  def getAnticipatedGhosts(self, gameState):
    anticipatedGhosts = []
    # Bayesian Inference Update Beliefs Function
    # ============================================================='
    for enemy in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(enemy)
      if not pos:
        self.elapseTime(enemy, gameState)
        self.observe(enemy, gameState)
      else:
        self.setTruePos(enemy, pos)

    for enemy in self.getOpponents(gameState):
      anticipatedPos = self.approxPos(enemy)
      enemyGameState = gameState.getAgentState(enemy) if anticipatedPos else None
      anticipatedGhosts.append((enemyGameState, anticipatedPos))

    return anticipatedGhosts
    # =============================================================

  def numberOfInvaders(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyHere = [a for a in enemies if a.isPacman]
    return len(enemyHere)

  ## Bayesian Inference Functions Ends.

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class MyOffensiveAgent(MyCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class MyDefensiveAgent(MyCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
