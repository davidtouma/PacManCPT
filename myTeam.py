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

from capture import GameState
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from collections import Counter
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MyOffensiveAgent', second = 'MyOffensiveAgent'):
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

###########
# Helpers #
###########

class Node:
  def __init__(self, position, parent=None):
    self.position = position
    self.parent = parent
    self.g = 0
    self.h = 0
    self.f = 0

paths = {}
state = "attack"
have_attacked = False


##########
# Agents #
##########

class MyCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions for offensive agent
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Defensive agent - states:
    Move to bottleneck, stay at bottleneck, intercept enemy agent, attack
    """
    if self.index == self.getTeam(gameState)[1]: # defensive agents
      global paths
      global state
      global have_attacked

      max_x, _ = max(gameState.data.layout.walls.asList(False))
      mid_x = int(max_x/2)
      
      if state == "move_to_bottleneck":
        current_position = gameState.getAgentPosition(self.index)
        current_goal = self.find_bottleneck(gameState)

        if not (current_position, current_goal) in paths: #check if the path has been calculated and stored before
          current_path = self.AStar(gameState.data.layout, current_position, current_goal) 
          paths[(current_position, current_goal)] = current_path
        else:
          current_path = paths[(current_position, current_goal)] #uses previous calculation if it exists

        # for cp in current_path:
        #   self.debugDraw(cp, (0,0,1))
        # self.debugDraw(current_goal, (1,0,1))

        if current_position == current_goal:
          state = "bottleneck"
          return "Stop"
        else:
          current_policy = self.get_policy(current_path) #converts the path into a policy (function that maps states to actions)
          return current_policy[current_position]


      if state == "bottleneck":
        if gameState.getAgentPosition(self.index) == gameState.getInitialAgentPosition(self.index):
          state = "move_to_bottleneck"
        enemy0_position = gameState.getAgentPosition(self.getOpponents(gameState)[0])
        enemy1_position = gameState.getAgentPosition(self.getOpponents(gameState)[1])

        if self.red and have_attacked == False:
          if (enemy0_position != None and enemy1_position != None) and (enemy0_position[0] < mid_x and enemy1_position[0] < mid_x):
            state = "attack"
        elif not self.red and have_attacked == False:
          if (enemy0_position != None and enemy1_position != None) and (enemy0_position[0] > mid_x+1 and enemy1_position[0] > mid_x+1):
            state = "attack"
        
        return "Stop"

      if state == "intercept":
        pass

      if state == "attack":
        have_attacked = True
        if gameState.getAgentPosition(self.index) == gameState.getInitialAgentPosition(self.index):
          state = "move_to_bottleneck"
        # if self.red:
        #   if gameState.getAgentPosition(self.index)[0] < mid_x:
        #     state = "move_to_bottleneck"
        # elif not self.red:
        #   if gameState.getAgentPosition(self.index)[0] > mid_x+1:
        #     state = "move_to_bottleneck"
        
        actions = gameState.getLegalActions(self.index)
    
        values = [self.evaluate(gameState, a) for a in actions]
        
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        self.numCarrying = gameState.data.agentStates[self.index].numCarrying

        is_Pacman = gameState.getAgentState(self.index).isPacman
        if is_Pacman: 
          if self.red:
            capsules = gameState.getBlueCapsules()
            enemy_idx = gameState.getBlueTeamIndices()

          elif not self.red: 
            capsules = gameState.getRedCapsules()
            enemy_idx = gameState.getRedTeamIndices()
          
          isCapsClose, caps_pos = self.is_caps_close(gameState, capsules, 8)
          scaredTimer = gameState.data.agentStates[enemy_idx[0]].scaredTimer
          
          if isCapsClose and scaredTimer <= 2:
            #print("Take capsuel")
            capsPath = self.AStar(gameState.data.layout, gameState.getAgentPosition(self.index), caps_pos)
            policy = self.get_policy(capsPath)
            return policy[gameState.getAgentPosition(self.index)]
          
          #Check scared timer och distance to home. Om scared timer > distance to home. Keep collecting food
          close_home_pos, dist_to_home = self.closest_dist_to_home(gameState)
          if scaredTimer > dist_to_home:
            #print("take food")
            return random.choice(bestActions)
          
          else:
            if self.numCarrying >= 5 or foodLeft <= 2:
              #print("return home")
              pathHome = self.AStar(gameState.data.layout, gameState.getAgentPosition(self.index), close_home_pos)
              '''
              for cp in pathHome:
                self.debugDraw(cp, (1,0,0))
              '''
              homePolicy = self.get_policy(pathHome)
              state = "move_to_bottleneck"
              return homePolicy[gameState.getAgentPosition(self.index)]
      
        return random.choice(bestActions)
        
    """
    Offensive agent:
    Picks among the actions with the highest Q(s,a).
    """
    if self.index == self.getTeam(gameState)[0]:
      actions = gameState.getLegalActions(self.index)
    
      values = [self.evaluate(gameState, a) for a in actions]
      
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      foodLeft = len(self.getFood(gameState).asList())

      self.numCarrying = gameState.data.agentStates[self.index].numCarrying

      ##Always offensive agent
      is_Pacman = gameState.getAgentState(self.index).isPacman
      if is_Pacman: 
          #Agent that always is the attacker:
        if self.red:
          capsules = gameState.getBlueCapsules()
          enemy_idx = gameState.getBlueTeamIndices()

        elif not self.red: 
          capsules = gameState.getRedCapsules()
          enemy_idx = gameState.getRedTeamIndices()
        
        isCapsClose, caps_pos = self.is_caps_close(gameState, capsules, 8)
        scaredTimer = gameState.data.agentStates[enemy_idx[0]].scaredTimer
        
        if isCapsClose and scaredTimer <= 2:
          #print("Take capsuel")
          capsPath = self.AStar(gameState.data.layout, gameState.getAgentPosition(self.index), caps_pos)
          policy = self.get_policy(capsPath)
          return policy[gameState.getAgentPosition(self.index)]
        
        #Check scared timer och distance to home. Om scared timer > distance to home. Keep collecting food
        close_home_pos, dist_to_home = self.closest_dist_to_home(gameState)
        if scaredTimer > dist_to_home:
          #print("take food")
          return random.choice(bestActions)
        
        else:
          if self.numCarrying >= 5 or foodLeft <= 2:
            #print("return home")
            pathHome = self.AStar(gameState.data.layout, gameState.getAgentPosition(self.index), close_home_pos)
            '''
            for cp in pathHome:
              self.debugDraw(cp, (1,0,0))
            '''
            homePolicy = self.get_policy(pathHome)
            return homePolicy[gameState.getAgentPosition(self.index)]
    
      return random.choice(bestActions)

  def is_caps_close(self, gameState, capsPositions, threshold):
    dist = math.inf
    for capsule in capsPositions:
      distToCaps = self.getMazeDistance(gameState.getAgentPosition(self.index), capsule)
      if distToCaps < dist:
        dist = distToCaps
        caps_pos = capsule
    if dist < threshold:
      return True, caps_pos
    else:
      return False, None
  
  def closest_dist_to_home(self, gameState):
    layout = gameState.data.layout
    nodes = layout.walls.asList(False)
    max_x, max_y = max(nodes)
    min_x, min_y = min(nodes)

    mid_x = int(max_x/2)
    if not self.red:
      mid_x += 1
    
    mid_point = (mid_x, min_y)

    dist = math.inf
    for i in range(1, max_y -1):
      
      if layout.isWall(mid_point) == True:
        continue

      distToHome = self.getMazeDistance(gameState.getAgentPosition(self.index), mid_point)
      if distToHome < dist:
        dist = distToHome
        home_pos = mid_point
      mid_point = (mid_x, i)
    return home_pos, dist

  def find_bottleneck(self, gameState):
    layout = gameState.data.layout
    
    food_to_defend = self.getFoodYouAreDefending(gameState).asList(True)
    capsules_to_defend = self.getCapsulesYouAreDefending(gameState)

    entry_points = layout.walls.asList(False)
    max_x, max_y = max(entry_points)
    min_x, min_y = min(entry_points)
    mid_x = int(max_x/2)

    paths = []
    paths_collected = []

    gameState.getInitialAgentPosition(self.index)
    friendly_start = gameState.getInitialAgentPosition(self.getTeam(gameState)[0])
    enemy_start = gameState.getInitialAgentPosition(self.getOpponents(gameState)[0])
    if not self.red:
      friendly_start, enemy_start = enemy_start, friendly_start
      mid_x += 1
    initial_path = self.AStar(layout, enemy_start, friendly_start)

    start_path = [x for x in initial_path if x[0] == mid_x]

    initial_start = start_path[0]
    initial_starts = [initial_start]
    for y in range(-3, 3+1):
      child = (initial_start[0], self.clamp(initial_start[1] + y, min_y, max_y))
      if not layout.isWall(child):
        initial_starts.append(child)

    for iss in initial_starts:
      for capsule in capsules_to_defend:
        path = self.AStar(layout, iss, capsule)
        paths.append(path)
        paths_collected.extend(path)
      for food in food_to_defend:
        path = self.AStar(layout, iss, food)
        paths.append(path)
        paths_collected.extend(path)
      path_counter = Counter(paths_collected)
    
    pos_counter = [pos for pos, freq in path_counter.items() if freq == path_counter.most_common(1)[0][1]]
    bottleneck = pos_counter[0]

    return bottleneck

  def AStar(self, layout, start, end):
    #grid = layout.walls.asList(False)
    
    start_node = Node(start)
    end_node = Node(end)

    open = []
    closed = []

    open.append(start_node)

    while len(open) > 0:
      current_node = open[0]
      current_index = 0
      for i in range(len(open)):
        if open[i].f < current_node.f:
          current_node = open[i]
          current_index = i
      
      del open[current_index]
      closed.append(current_node)

      if current_node.position == end_node.position:
        path = []
        while current_node != None:
          path.append(current_node.position)
          current_node = current_node.parent
        path.reverse()
        return path
      
      children = []
      xy_moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
      for xy in xy_moves:
        x, y = xy
        child_position = (current_node.position[0] + x, current_node.position[1] + y)

        if x==0 and y==0:
          continue
        
        if layout.isWall(child_position):
          continue
        
        new_node = Node(child_position, current_node)
        children.append(new_node)

      for child in children:
        if child in closed:
          continue
        
        child.g = current_node.g + 1
        child.h = self.getMazeDistance(child.position, end_node.position)
        child.f = child.g + child.h

        if child in open:
          child_index = open.index(child)
          if child.g > open[child_index].g:
            continue
          else:
            del open[child_index]
        
        open.append(child)
    
    return []

  def get_policy(self, path):
    actions = []
    i = 0
    while i < len(path)-1:
      x1, y1 = path[i]
      x2, y2 = path[i+1]
      direction = (x2-x1, y2-y1)
      if direction == (1, 0):
        action = "East"
      if direction == (0, 1):
        action = "North"
      if direction == (-1, 0):
        action = "West"
      if direction == (0, -1):
        action = "South"
      if direction == (0, 0):
        action = "Stop"
      i += 1
      actions.append(action)

      policy = dict(zip(path, actions))

    return policy
      
  def clamp(self, n, smallest, largest):
    return max(smallest, min(n, largest))

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
    print("features:", features)
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
"""
