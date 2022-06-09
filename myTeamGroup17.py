# myTeam.py
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

from operator import pos
from turtle import position

from pkg_resources import load_entry_point
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from distanceCalculator import Distancer

import math

#################
# Team creation #
#################

## Node for the position tree
class Node:
    def __init__(self, position, heuristic_value, parent = None, move = None):
        self.position = position
        self.heuristic_value = heuristic_value
        self.children = []
        self.parent = parent
        self.move_to_get_here = move

    def add_child(self, obj):
        self.children.append(obj)
    
    def get_children(self):
        return self.children
      

def get_leaf_nodes(root):
    leafs = []
    def _get_leaf_nodes( node):
        if node is not None:
            if len(node.get_children()) == 0:
                leafs.append(node)
            for n in node.get_children():
                _get_leaf_nodes(n)
    _get_leaf_nodes(root)
    return leafs

# Returns true if there is a path from
# root to the given node. It also
# populates 'arr' with the given path
# x is the searched for node
def hasPath(root, arr_of_pos, x, arr_of_nodes):
    if (not root):
        return False
     
    arr_of_pos.append(root.position)
    arr_of_nodes.append(root)
    if (root == x):    
        return True
    
    for child in root.get_children():
        if(hasPath(child, arr_of_pos, x, arr_of_nodes)):
            return True

    arr_of_pos.pop(-1)
    arr_of_nodes.pop(-1)
    return False


def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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

def maxDepth(node):
  # No children means depth zero below.

  if len(node.children) == 0:
    return 0

  # Otherwise get deepest child recursively.

  deepestChild = 0
  for child in node.children:
      childDepth = maxDepth(child)
      if childDepth > deepestChild:
          deepestChild = childDepth

  # Depth of this node is one plus the deepest child.
  return 1 + deepestChild


def layers_to_root(root_node, curr, depth):
    if root_node == curr:
        return depth
    else:
        depth += 1
        return layers_to_root(root_node, curr.parent, depth)

##########
# Agents #
###########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    # print("gameState: \n", gameState)
  
  def create_tree_for_actions(self, gameState, root_node, current_node, num_layers):
    '''
    Creates tree with a certain depth, does not return anything as the memory of the root_node is stored
    '''
    if(layers_to_root(root_node, current_node, 0) > num_layers):
        return

    else:
        actions = gameState.getLegalActions(self.index)

        for next_move in actions: # search one layer
            if next_move != "Stop":
                successor_gamestate = self.getSuccessor(gameState, next_move)
                child_pos = successor_gamestate.getAgentPosition(self.index)
                child_heuristic = self.evaluate(gameState, next_move)
                child_node = Node(child_pos, child_heuristic, parent=current_node, move=next_move)
                current_node.add_child(child_node)

                self.create_tree_for_actions(successor_gamestate, root_node, child_node, num_layers)


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # values = [self.evaluate(gameState, a) for a in actions]


    ## Build tree of actions
    root_node = Node(gameState.getAgentPosition(self.index), self.evaluate(gameState, "Stop"))

    number_of_layers_in_tree = 4

    for next_move in actions: # search one layer  ## REMOVE THE STOP COMMAND?
        if next_move != "Stop":
            # print("next move: ", next_move)
            successor_gamestate = self.getSuccessor(gameState, next_move)
            child_pos = successor_gamestate.getAgentPosition(self.index)
            child_heuristic = self.evaluate(gameState, next_move)
            child_node = Node(child_pos, child_heuristic, parent=root_node, move=next_move)
            root_node.add_child(child_node)

            self.create_tree_for_actions(successor_gamestate, root_node, child_node, number_of_layers_in_tree) # and then kick up the automatic tree creation

    ## END BUILDING TREE

    ## CHECK BEST LEAF NODE OF TREE
    leaf_nodes = get_leaf_nodes(root_node)

    maxValue = -math.inf
    maxNode = None
    bestPath = []
    bestNodes = []
    for leaf in leaf_nodes:
        arr = []
        arr_of_nodes = []
        hasPath(root_node, arr, leaf, arr_of_nodes)
        # print(arr)
        heuristic_value_list = [x.heuristic_value for x in arr_of_nodes[1:]]
        sum_of_heuristic_values = sum(heuristic_value_list)

        # print("heuristic_value_list: ", heuristic_value_list)
        # print("sum_of_heuristic_values: ", sum_of_heuristic_values)
        if(sum_of_heuristic_values > maxValue):
            maxNode = leaf
            bestPath = arr.copy()
            bestNodes = arr_of_nodes.copy()
            maxValue = sum_of_heuristic_values
            

    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    ## TODO: Perhaps change heuristics to infinity?
    
    # find possible hole
    possible_hole_list = self.getPossibleHole(gameState)
    # find enemy index
    if not self.red: # we are in the blue team
        enemy_team_indices = gameState.getRedTeamIndices()
    else:
        enemy_team_indices = gameState.getBlueTeamIndices()    
    
    enemyPos1 = gameState.getAgentPosition(enemy_team_indices[0])
    enemyPos2 = gameState.getAgentPosition(enemy_team_indices[1])
    currentPos = gameState.getAgentPosition(self.index)
    
    nearestEnemeyPos = None
    if enemyPos1 != None:
        nearestEnemeyPos = enemyPos1
    if enemyPos2 != None:
        if enemyPos1 != None:
            if self.getMazeDistance(currentPos,enemyPos1) > self.getMazeDistance(currentPos,enemyPos2):
                nearestEnemeyPos = enemyPos2
        else:
            nearestEnemeyPos = enemyPos2
    nearest_enemy_distance = 999
    if nearestEnemeyPos != None:
        nearest_enemy_distance = self.getMazeDistance(currentPos,nearestEnemeyPos)
    
    agentState = gameState.data.agentStates[self.index]
    carried_food = agentState.numCarrying
    if carried_food >= 3 and nearest_enemy_distance > 3: # Return home when have 3 food on you!
      print("RETURN!")
      bestDist = 9999
      besthole = possible_hole_list[0]
      for hole in possible_hole_list:
          if self.getMazeDistance(gameState.getAgentPosition(self.index), hole) < self.getMazeDistance(gameState.getAgentPosition(self.index), besthole):
              besthole = hole 
    #   self.debugDraw(besthole,[0,1,1])           
      for action in actions:
        successor = self.getSuccessor(gameState, action)            
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(besthole,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction   
  
    # self.debugDraw(bestNodes[-1].position,[0,1,0],clear = True)
    # print(actions)   
    return bestNodes[1].move_to_get_here
  
  def getMapOffset(self,gameState):
    half_width = (gameState.getWalls().width - 2)/2
    half_height = (gameState.getWalls().height - 2)/2
    return (half_width,half_height)
  
  def getPossibleHole(self,gameState): # compute the closest possible hole 
    possible_hole_list = []
    (half_width,half_height) = self.getMapOffset(gameState)
    if not self.red: # we are in the blue team
        offset = self.start[0] - half_width + 1
    else:
        offset = self.start[0] + half_width - 1
        
    for i in range(gameState.getWalls().height-2):
        if not gameState.hasWall(int(offset),i+1):
            possible_hole = (offset,i+1)
            possible_hole_list.append(possible_hole)
    return possible_hole_list

  def getPossibleMove(self,gamestate,x,y): # check possible move
      move_list = []
      if not gamestate.hasWall(x,y+1):
          move_list.append('North')
      if not gamestate.hasWall(x,y-1):
          move_list.append('South')
      if not gamestate.hasWall(x+1,y):
          move_list.append('East')
      if not gamestate.hasWall(x-1,y):
          move_list.append('West')
      return move_list     
  
      
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

class OffensiveReflexAgent(ReflexCaptureAgent):
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

    
    
    ########## Go to the safe hole START ##################
    possible_hole_list = self.getPossibleHole(gameState)
    (half_width,half_height) = self.getMapOffset(gameState)
    currentPos = gameState.getAgentPosition(self.index)
    
    # Go to the safe hole (nearest to the food) before crossing the boundart
    InOurArea = False
    if not self.red: # we are in the blue team
        if currentPos[0] > half_width: 
            InOurArea = True            
    else:
        if currentPos[0] < half_width + 1:
            InOurArea = True  
                        
    if InOurArea == True:
        for hole in possible_hole_list:
            minDistance = min([self.getMazeDistance(hole, food) for food in foodList])
            # print("Go to safe hole")
        features['distanceToHole'] = minDistance              
    ########## Go to the safe hole END ##################
    
    ############ Enemy avoidance heuristic START ###########
    if not self.red: # we are in the blue team
        enemy_team_indices = gameState.getRedTeamIndices()
    else:
        enemy_team_indices = gameState.getBlueTeamIndices()    
    
    enemyPos1 = gameState.getAgentPosition(enemy_team_indices[0])
    enemyPos2 = gameState.getAgentPosition(enemy_team_indices[1])
    nearestEnemeyPos = None
    if enemyPos1 != None:
        nearestEnemeyPos = enemyPos1
    if enemyPos2 != None:
        if enemyPos1 != None:
            if self.getMazeDistance(myPos,enemyPos1) > self.getMazeDistance(myPos,enemyPos2):
                nearestEnemeyPos = enemyPos2
        else:
            nearestEnemeyPos = enemyPos2
    
    if nearestEnemeyPos != None:
        nearest_enemy_distance = self.getMazeDistance(myPos,nearestEnemeyPos)
        if(nearest_enemy_distance <= 3):
            # print("Enemy close by")
            features['closest_enemy_distance'] = nearest_enemy_distance
            move_list = self.getPossibleMove(gameState,currentPos[0],currentPos[1])
            if len(move_list) <= 1:
                features['Deadend'] = 1    
    ############ Enemy avoidance heuristic END ###########

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, "closest_enemy_distance":-100, 'distanceToHole': -100, 'Deadend': -1000}



def get_highest_and_lowest_points(pos_list):
    ''' Returns highest and lowest points given (on the y-axis)
    '''
    highest_point = pos_list[0]
    lowest_point = pos_list[0]
    # print("pos_list[0]: ", pos_list[0])
    # print("pos_list[0]: ", pos_list[0][1])
    highest_value = pos_list[0][1]
    lowest_value = pos_list[0][1]

    for point in pos_list:
        if point[1] > highest_value:
            highest_point = point
            highest_value = point[1]
        
        if point[1] < lowest_value:
            lowest_point = point
            lowest_value = point[1]
    
    return highest_point, lowest_point


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    # print("Def agent")
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    ## START Heuristics for putting the def agent in defensive position
    goal_point = None
    if not self.red: # BLUE TEAM
        our_team_food_list = gameState.getBlueFood().asList()
        enemy_team_indices = gameState.getRedTeamIndices() # as we are in the blue team
        enemy_spawn_point = gameState.getInitialAgentPosition(enemy_team_indices[0])
        

    else: # RED TEAM
        our_team_food_list = gameState.getRedFood().asList()
        enemy_team_indices = gameState.getBlueTeamIndices() # as we are in the red team
        enemy_spawn_point = gameState.getInitialAgentPosition(enemy_team_indices[0])

    
    distances_to_food_list = ([self.getMazeDistance(enemy_spawn_point, food) for food in our_team_food_list])
    index_min = min(range(len(distances_to_food_list)), key=distances_to_food_list.__getitem__)
    closest_food_to_enemy_spawn = our_team_food_list[index_min]

    goal_point = closest_food_to_enemy_spawn
    features["distance_to_choke_point"] = self.getMazeDistance( myPos, goal_point )
    #### END DEFENSIVE POSITION

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
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distance_to_choke_point': -99}
    # return {'numInvaders': 0, 'onDefense': 0, 'invaderDistance': 0, 'stop': 0, 'reverse': 0, 'distance_to_choke_point': -9999999}
