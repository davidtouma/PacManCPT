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


from __future__ import annotations
from distutils.log import debug

from math import inf, exp
from re import M
from time import sleep
from typing import Tuple
from capture import GameState
from captureAgents import CaptureAgent

import random, util
from game import AgentState, Actions
import game

GO_BACK_PERCENTAGE_THRESHOLD = 0.5
MAX_COUNTER = 300
MAX_CHASING_TIME = 50
MAX_CAPSULE_TIMEOUT = 30

class Inlet:
  def __init__(self, start_pos, end_pos):
    self.start_pos = start_pos
    self.end_pos = end_pos
    self.size = util.manhattanDistance(start_pos, end_pos)
  def get_center(self):
    return (int ((self.start_pos[0] + self.end_pos[0]) / 2), int((self.start_pos[1] + self.end_pos[1]) / 2))

Position = Tuple[int, int]

class Node():
    """A node class for A* Pathfinding"""
    
    parent : None | Node = None 
    position : Position = None
    g = 0
    h = 0
    f = 0

    def __init__(self, parent : None | Node = None, position : Position = None):
        self.parent = parent
        self.position = position

    def __eq__(self, other : Node) -> bool:
        return self.position == other.position
    def __hash__(self):
        return hash(self.position)



def path_to_moves(path):
  """
  Given a path (a list of nodes), return a list of moves.
  """
  moves = []
  for i in range(len(path)-1):

    vector = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
    direction = Actions.vectorToDirection(vector)
    moves.append(direction)
  return moves

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Attacker', second = 'Defender'):
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

class MainAgent(CaptureAgent) :
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState : GameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    #for line in self.getFood(gameState):
    ##  ###print('  '.join(map(str, line)))
    ####print(self.getTeam(gameState))
    

    '''
    Your initialization code goes here, if you need any.
    '''

    self.walls = gameState.getWalls().asList()
    self.wallsDict = {wallPos: True for wallPos in self.walls}
    self.walls = gameState.getWalls().data
    self.layout = gameState.data.layout
    self.startState = gameState.getAgentPosition(self.index)
    self.isOnRedTeam = gameState.isOnRedTeam(self.index)
    self.wasPacman = False
    self.inlets = []
    walls = gameState.getWalls()
    self.counter = -1

    if(self.isOnRedTeam):
      self.middle_index = int( self.layout.width / 2)-1
    else:
      self.middle_index = int( self.layout.width / 2)
    
    current_inlet_start_y = 0
    for i in range(self.layout.height):
      if(walls[self.middle_index][current_inlet_start_y]):
        current_inlet_start_y += 1
        i = current_inlet_start_y
      if(walls[self.middle_index][i]):
        self.inlets.append(Inlet((self.middle_index, current_inlet_start_y), (self.middle_index, i-1)))
        current_inlet_start_y = i+1

    """for x in self.wallsDict.keys():
      self.debugDraw([x], [0,0,1])"""
    
    for inlet in self.inlets:
      for i in range(inlet.start_pos[1], inlet.end_pos[1]+1):
        break
        self.debugDraw([(inlet.start_pos[0], i)], [1,1,0] if self.isOnRedTeam else [0,1,1])

    ##print("The field sixe is %d %d" % (self.layout.width, self.layout.height))



  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    #actions = gameState.getLegalActions(self.index)
    ####print("choose action")

    ####print(self.getFood(self, gameState))

    '''
    You should change this in your own agent.
    '''
    #return random.choice(actions)

  #target is a 1x2 matrix with the point to go to, e.g. [9,12]


# Function has to be here such that we can reference 'MainAgent'

def astar(maze : list[list[bool]], start : Position, end : Position, agent: MainAgent, return_cost: bool = False, is_invincible: bool = False):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    ###print("going from node " + str(start) + " to node " + str(end))

    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)
    
    # Initialize both open and closed list
    open_list = set()
    closed_list = set()

    # Add the start node
    open_list.add(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        ###print("open list length: " + str(len(open_list)))

        # Get the current node
        current_node = random.sample(open_list, 1)[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.remove(current_node)
        #open_list.pop(current_index)
        closed_list.add(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] if not return_cost else (path[::-1], current_node.f)# Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[(int)(node_position[0])][int(node_position[1])]:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child in closed_list:
                continue

            #for closed_child in closed_list:
             #   if child == closed_child:
              #      continue

            # Create the f, g, and h values
            child.g = current_node.g + position_cost(current_node, agent, is_invincible)
            # We could also use 'agent.getMazeDistance' if we had access to that
            child.position = (int (child.position[0]), int (child.position[1]))
            child.h = agent.getMazeDistance(child.position, end_node.position)
            
            #child.h = calculate_h(child,end_node)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.add(child)


# This function is used to determine the punishment weight of 
# the given position
# A pacman for example should not run into a ghost if it is not scared
# also power ups should probably be a priority while choosing a path
# next priority should be food. 
def position_cost(current_node : Node, agent : MainAgent, invincible: bool = False):
  gameState : GameState = agent.getCurrentObservation()
  myState : AgentState = gameState.getAgentState(agent.index)
  observable_enemies = list(filter(
    lambda pos : pos != None,
      map(
        lambda index : gameState.getAgentPosition(index),
        agent.getOpponents(gameState)
      )
      ))
  node_is_enemy = current_node.position in observable_enemies

  if(myState.isPacman ):
    node_is_food = agent.getFood(gameState)[int(current_node.position[0])][int(current_node.position[1])]
    node_is_powerup = current_node.position in agent.getCapsules(gameState)

    # TODO ajust magic numbers
    if node_is_enemy:
      return inf if (myState.isPacman and not invincible) or (not myState.isPacman and myState.scaredTimer == 0) else 0 # Please don't hurt me :( (but only if you are scary)
    elif node_is_food:
      return 2.5 # Yummy
    elif node_is_powerup:
      return 1 # even more yummy
    
    return 10 # We could make this dynamic based on the amount of food left

  else: # We are a defender 
    if node_is_enemy:
      return 0
    if(not agent.isAttacker):
      if(agent.isOnRedTeam):
        return 100 if current_node.position[0] > agent.middle_index else 1
      else:
        return 100 if current_node.position[0] < agent.middle_index else 1
    return 1

def find_target_attack_area( agent : Defender, eaten_food: list[Position]):
  
  gameState : GameState = agent.getCurrentObservation()
  myState : AgentState = gameState.getAgentState(agent.index)

  if(len(eaten_food)>0): #faccio un tradeoff in base a distanza e score del punto mangiato
    max_cost = inf
    final_path = None
    for food in eaten_food:
      food_pos = (int(food[0]), int(food[1]))
      path, cost = astar(agent.walls, myState.getPosition(), food_pos, agent, True)
      score = agent.score_map[food_pos[0]][food_pos[1]]
      if(score == 0):
        agent.score_map = agent.compute_score_map(gameState)
        score = agent.score_map[food_pos[0]][food_pos[1]]
        if(score == 0): #this means the eaten food was alone in the map and it is not worth to go there
          score = 1
      #print("E' stato mangiato cibo in posizione: ", (food_pos), " con score ", score)
      #######sleep(0.1)
      if(cost/score<max_cost):
        max_cost = cost/score
        final_path = path
    agent.estimated_enemy_position = final_path[-1]
    #agent.debugDraw(agent.estimated_enemy_position, [1,0,0], True)
    return agent.estimated_enemy_position
  else: #Aggiungere memoria su dove credo che ci sia il pacman
    if(agent.estimated_enemy_position is not None and agent.chasing_step < MAX_CHASING_TIME):
      agent.chasing_step += 1
      return agent.estimated_enemy_position
    else:
      agent.chasing_step = 0
      agent.estimated_enemy_position = None
      return agent.path_to_biggest_inlet(gameState)[-1]


class Defender(MainAgent):
      
  def compute_score_map(self, gameState):
    ###print("Computing score map")
    if(not self.isOnRedTeam):
      food = gameState.getBlueFood()
      capsule = gameState.getBlueCapsules()
    else:
      food = gameState.getRedFood()
      capsule = gameState.getRedCapsules()

    score_map = []

    for i in range(0, self.layout.width):
      score_map.append([])
      for j in range(0, self.layout.height):
        score_map[i].append(0)
    for i in range(0, self.layout.width):
      for j in range(0, self.layout.height):
        if((i,j) in capsule):
          score_map[i][j] = 1
          food[i][j] = True
        if(food[i][j]):
          score_map[i][j] += 1
          for nearbypos in [(0, -1), (0, 1), (-1, 0), (1, 0), (1,-1), (-1, 1), (1,1), (-1,-1)]: # Adjacent squares
          # Get node position
            (new_i, new_j) = (i + nearbypos[0], j + nearbypos[1])
            if(new_i>=0 and new_i<self.layout.width and new_j>=0 and new_j<self.layout.height):
              if(food[new_i][new_j]):
                score_map[i][j] += 1
    """for i in range(0, self.layout.width):
      for j in range(0, self.layout.height):
        if(score_map[i][j]):
          for nearbypos in [(0, -1), (0, 1), (-1, 0), (1, 0), (1,-1), (-1, 1), (1,1), (-1,-1)]: # Adjacent squares
            (new_i, new_j) = (i + nearbypos[0], j + nearbypos[1])
            if(new_i>=0 and new_i<self.layout.width and new_j>=0 and new_j<self.layout.height):
              #break
              score_map[i][j] += score_map[new_i][new_j]*0.5
              if(not food[i][j]):
                score_map[i][j]=0
                """
    return score_map

  def path_to_closest_inlet(self, gameState):
    myState = gameState.getAgentState(self.index)
    path = None
    for inlet in self.inlets:
      inlet_middle = inlet.get_center()
      path_to_inlet = astar(self.walls, myState.getPosition(), inlet_middle, self)
      if(path is None or len(path)>len(path_to_inlet)):
        path = path_to_inlet
    return path

  def path_to_biggest_inlet(self, gameState):
    myState = gameState.getAgentState(self.index)
    path = None
    inlet_size = 0
    inlet_center = None
    for inlet in self.inlets:
      if(inlet.size>inlet_size):
        inlet_size = inlet.size
        inlet_center = inlet.get_center()
    if(inlet_center is not None):
      path = astar(self.walls, myState.getPosition(), inlet_center, self)
    return path
  
  def is_enemy_in_my_field(self, enemy_position: Position):
    if(self.isOnRedTeam):
      return enemy_position[0]<=self.middle_index
    else:
      return enemy_position[0]>=self.middle_index

  def registerInitialState(self, gameState):
    MainAgent.registerInitialState(self, gameState)
    self.isAttacker = False
    #self.debugDraw([(33,17)], [1,0,0])
    self.path = []
    #if(self.index==3):
     # self.path = []#astar(gameState.getWalls().data, gameState.getAgentState(self.index).getPosition(),gameState.getAgentState(self.index-1).getPosition())
      ###print(self.path)
      #for i in range(len(self.path)):
       # self.debugDraw([self.path[i]], [0,1,0])
    ##print(path_to_moves(self.path))
    ##print("I am a Defender ", self.index, "at position ", gameState.getAgentState(self.index).getPosition())
    ##print("ATTENTION REQUIRED ON LINE 354")
    self.move_index = 0
    self.previousPosition = gameState.getAgentState(self.index).getPosition()
    self.score_map = self.compute_score_map(gameState)
    self.estimated_enemy_position:Position = None
    self.chasing_step = 0



  def chooseAction(self, gameState):
      ##Impelemtn attacker action

    myState = gameState.getAgentState(self.index)
    actions = gameState.getLegalActions(self.index)
    eaten = [] #list containing the position where food has been eaten

    ###print("I'm agent ", self.index, " at position ", myState.getPosition())

    previous_observation = self.getPreviousObservation()
    if previous_observation is None:
      previous_observation = gameState


    if(not self.isOnRedTeam):
      food = gameState.getBlueFood()
      old_food = previous_observation.getBlueFood()
    else:
      food = gameState.getRedFood()
      old_food = previous_observation.getRedFood()
    
    for i in range(0, food.width):
      for j in range(0, food.height):
        if(not food[i][j] and old_food[i][j]):
          #self.debugDraw([(i,j)], [1,0,0])
          eaten.append((i,j)) #list containing the position where food has been eaten by the opponent pacman

    if(not myState.isPacman): #if ghost, chase an opponent if possible
      
      opponents_pos = [(x,gameState.getAgentPosition(x)) for x in self.getOpponents(gameState)]
      opponents_pos = [(x,y) for (x,y) in opponents_pos if y is not None]
      is_opponent_attacker = False
      for opponent in opponents_pos:
        if(self.is_enemy_in_my_field(opponent[1])):
          is_opponent_attacker=True
          break
      # and not (not is_opponent_attacker and self.estimated_enemy_position is not None)
      if(len(opponents_pos)>0 and (is_opponent_attacker or (len(eaten)==0 and self.estimated_enemy_position is None))):
        
        self.estimated_enemy_position = None
        self.chasing_step = 0

        closest_opponent = min(opponents_pos, key=lambda x: self.getMazeDistance(myState.getPosition(), x[1])* 1 if self.is_enemy_in_my_field(x[1]) else 100)
        x = closest_opponent[1][0]
        y = closest_opponent[1][1]
        target= (x,y)
        self.path = astar(self.walls, myState.getPosition(), target , self)
      else: #if no opponent is visible, do infer the best position to go
        self.path = astar(self.walls, myState.getPosition(), find_target_attack_area(self, eaten) , self)
        ##print("Vado nell'area di attacco")
      #self.path = astar(self.walls, gameState.getAgentState(self.index).getPosition(), find_target_attack_area(self), self)
      #self.debugDraw(self.path, [0,1,0], True)
            
      if(len(eaten)>0):
        self.score_map = self.compute_score_map(gameState)

      moves = path_to_moves(self.path)
      
      if(len(moves) == 0):
        ###print("Fine")
        return "Stop"
      if(self.isOnRedTeam and self.path[1][0]> self.middle_index or not self.isOnRedTeam and self.path[1][0]< self.middle_index):
        return "Stop"

      if moves[0] not in gameState.getLegalActions(self.index):
        ##print("Azione illegale")
        return "Stop"
      return moves[0]
    
    else: #if pacman, go to the closest inlet WARNING WE NEVER ENTER HERE BECAUSE OF THE IF ABOVE
            
      if(len(eaten)>0):
        self.score_map = self.compute_score_map(gameState)

      ##print("Go to the closest inlet and enter your field") #lo sto facendo andare al centro del campo
      path = None
      for inlet in self.inlets:
        inlet_middle = inlet.get_center()
        path_to_inlet = astar(self.walls, myState.getPosition(), inlet_middle, self)
        if(path is None or len(path)>len(path_to_inlet)):
          path = path_to_inlet

      moves = path_to_moves(path)
      if(len(moves) == 0):
        ###print("Fine")
        return "Stop"
      if moves[0] not in gameState.getLegalActions(self.index):
        ##print("Azione illegale")
        return "Stop"
      return moves[0]
  

class Attacker(MainAgent):

  def has_eaten_ghost(self, gameState : GameState):

    myState : AgentState = gameState.getAgentState(self.index)
    opponents = self.getOpponents(gameState)
    to_return = False
    for opponent in opponents:
      opponent_state = gameState.getAgentState(opponent)
      if (not opponent_state.isPacman and opponent_state.scaredTimer != 0):
        to_return = True
        break
    return to_return


    if(not myState.isPacman):
      return False
    else:
      opponents = [(x,gameState.getAgentPosition(x)) for x in self.getOpponents(gameState)]
      opponents = [(x,y) for (x,y) in opponents if y is not None]
      opponents_distances = [(x, self.getMazeDistance(myState.getPosition(),y)) for (x,y) in opponents]

      for (x,y) in opponents_distances:
        if(y<=2):
          self.close_agents.append(x)
      for agent in self.close_agents:
        opponents = [(x,gameState.getAgentPosition(x)) for x in self.getOpponents(gameState)]
        opponents = [ x for (x,y) in opponents if y is not None]
        if(agent not in opponents):
          self.close_agents.remove(agent)
          return True
      return False


  def compute_score_map(self, gameState):
    ###print("Computing score map")
    if(self.isOnRedTeam):
      food = gameState.getBlueFood()
      capsule = gameState.getBlueCapsules()
    else:
      food = gameState.getRedFood()
      capsule = gameState.getRedCapsules()

    score_map = []

    for i in range(0, self.layout.width):
      score_map.append([])
      for j in range(0, self.layout.height):
        score_map[i].append(0)
    for i in range(0, self.layout.width):
      for j in range(0, self.layout.height):
        if((i,j) in capsule):
          score_map[i][j] = 2
          food[i][j] = True
        if(food[i][j]):
          score_map[i][j] = 1
          for nearbypos in [(0, -1), (0, 1), (-1, 0), (1, 0), (1,-1), (-1, 1), (1,1), (-1,-1)]: # Adjacent squares
          # Get node position
            (new_i, new_j) = (i + nearbypos[0], j + nearbypos[1])
            if(new_i>=0 and new_i<self.layout.width and new_j>=0 and new_j<self.layout.height):
              score_map[i][j] += score_map[new_i][new_j]
    for i in range(0, self.layout.width):
      for j in range(0, self.layout.height):
        if(score_map[i][j]):
          for nearbypos in [(0, -1), (0, 1), (-1, 0), (1, 0), (1,-1), (-1, 1), (1,1), (-1,-1)]: # Adjacent squares
            (new_i, new_j) = (i + nearbypos[0], j + nearbypos[1])
            if(new_i>=0 and new_i<self.layout.width and new_j>=0 and new_j<self.layout.height):
              #break
              score_map[i][j] += score_map[new_i][new_j]*0.5
              if(not food[i][j]):
                score_map[i][j]=0
    return score_map


  def registerInitialState(self, gameState):
    self.isAttacker = True
    MainAgent.registerInitialState(self, gameState)
    ##print("I am an Attacker")
    self.score_map = self.compute_score_map(gameState)
    self.last_eaten = 0
    self.target = None
    self.has_eaten_capsule = False
    self.remaining_capsules = 2
    self.capsule_timeout = MAX_CAPSULE_TIMEOUT
    self.close_agents = []
    for i in range (0, self.layout.width):
      ##print(self.score_map[i])
      pass

  
  def chooseAction(self, gameState):
      ##Impelemtn attacker action
    self.counter += 1

    if(self.isOnRedTeam):
      self.capsules = gameState.getBlueCapsules()
    else:
      self.capsules = gameState.getRedCapsules()

    if(len(self.capsules)<self.remaining_capsules):
      self.remaining_capsules = len(self.capsules)
      self.has_eaten_capsule = True
      self.capsule_timeout = MAX_CAPSULE_TIMEOUT   

    for opponent in self.getOpponents(gameState):
      if(gameState.getAgentState(opponent).scaredTimer == 0):
        self.has_eaten_capsule = False

    if(self.has_eaten_capsule):
      self.capsule_timeout -= 1
    if(self.capsule_timeout<=0):
      self.has_eaten_capsule = False

    myState = gameState.getAgentState(self.index)
    my_position = myState.getPosition()
    my_position = (int (my_position[0]), int (my_position[1]))
    actions = gameState.getLegalActions(self.index)
    carrying = myState.numCarrying
    if(self.target is None or self.target == myState.getPosition()):
      self.score_map = self.compute_score_map(gameState)
    eaten = []


   

    previous_observation = self.getPreviousObservation()
    if previous_observation is None:
      previous_observation = gameState

    if(self.isOnRedTeam):
      food = gameState.getRedFood()
      old_food = previous_observation.getRedFood()
    else:
      food = gameState.getBlueFood()
      old_food = previous_observation.getBlueFood()
    
    for i in range(0, food.width):
      for j in range(0, food.height):
        if(not food[i][j] and old_food[i][j]):
          #self.debugDraw([(i,j)], [1,0,0])
          eaten.append((i,j))

    ####print("Legal actions are %s" % actions)
    ####print("Possible actions are %s" % self.getPossibleActions(gameState.getAgentPosition(self.index)))

    if(myState.isPacman):
      if(not self.wasPacman):
        self.total_food = len(self.getFood(gameState).asList())
      
      if(carrying>2):
        path_to_inlet = None
        for inlet in self.inlets:
          if(abs(my_position[0] - inlet.get_center()[0])==1):
            inlet_distance = 1
            path_to_inlet = [my_position,(inlet.start_pos[0], my_position[1])]
            break
        if(path_to_inlet is not None):
          moves = path_to_moves(path_to_inlet)
          if(len(moves) == 0):
            ####print("Fine")
            return "Stop"
          if moves[0] not in gameState.getLegalActions(self.index):
            ###print("Azione illegale")
            return "Stop"
          return moves[0]

      ####print("I am pacman")
      if(carrying>self.total_food*(GO_BACK_PERCENTAGE_THRESHOLD) * (0.5 if self.counter/MAX_COUNTER>0.85 else 1) and not self.has_eaten_capsule):#*exp(self.counter/70)/10
        inlet_distance = inf
        path_to_inlet = []
        for inlet in self.inlets:
          if(abs(my_position[0] - inlet.get_center()[0])==1):
            inlet_distance = 1
            path_to_inlet = [my_position,(inlet.start_pos[0], my_position[1])]
            break
          path = astar(self.walls, my_position, (inlet.start_pos[0], int((inlet.start_pos[1] + inlet.end_pos[1])/2)) , self, False , self.has_eaten_capsule)
          if(len(path)<inlet_distance):
            inlet_distance = len(path)
            path_to_inlet = path
        moves = path_to_moves(path_to_inlet)
        if(len(moves) == 0):
          ####print("Fine")
          return "Stop"
        if moves[0] not in actions:
          ###print("Azione illegale")
          return "Stop"
        myState.wasPacman = myState.isPacman
  
        return moves[0]

      else:
          self.target =  None
          max_score = 0
          for i in range(0, self.layout.width):
            for j in range(0, self.layout.height):
              if(self.score_map[i][j]>max_score and self.walls[i][j]==False):
                self.target = (i,j)
                max_score = self.score_map[i][j]
          #self.debugDraw([self.target], [1,1,1])      
          ####print("Il mio target è", self.target)
          path = astar(self.walls, my_position, self.target , self, False , self.has_eaten_capsule)
          moves = path_to_moves(path)
          max_score_position = None
          max_score = 0

          observable_enemies = list(filter(
          lambda pos : pos != None,
            map(
              lambda index : gameState.getAgentPosition(index),
              self.getOpponents(gameState)
            )
            ))
          if(self.isOnRedTeam):
            food = gameState.getBlueFood()
            for pos in gameState.getBlueCapsules():
              food[pos[0]][pos[1]] = True
              self.score_map[pos[0]][pos[1]] +=2
          else:
            food = gameState.getRedFood()
            for pos in gameState.getRedCapsules():
              if(len(gameState.getRedCapsules())>1):
                self.score_map[pos[0]][pos[1]] +=2
                food[pos[0]][pos[1]] = True
          my_position = myState.getPosition()
          my_position = (int (my_position[0]), int (my_position[1]))
          for nearbypos in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            ###print("C'è food in ", (my_position[0]+nearbypos[0], my_position[1]+nearbypos[1]), ":", food[my_position[0]+nearbypos[0]][my_position[1]+nearbypos[1]])
            chased = False
            opponents_pos = [(x,gameState.getAgentPosition(x)) for x in self.getOpponents(gameState)]
            opponents_pos = [(x,y) for (x,y) in opponents_pos if y is not None]
            if(len(opponents_pos)>0 and not self.has_eaten_capsule):
              for enemy in opponents_pos:
                if (self.getMazeDistance(my_position, enemy[1])<4):
                  chased = True
                  break
                
            if(food[my_position[0]+nearbypos[0]][my_position[1]+nearbypos[1]] and not chased):
              score = self.score_map[my_position[0]+nearbypos[0]][my_position[1]+nearbypos[1]]
              if(score>max_score):
                max_score = score
                max_score_position = (my_position[0]+nearbypos[0], my_position[1]+nearbypos[1])
          if max_score_position is not None:
            ##print("FACCIO L'HILLING VERSO ", max_score_position)
            #####sleep(0.1)
            path = astar(self.walls, my_position, max_score_position , self, False , self.has_eaten_capsule)
            moves = path_to_moves(path)
            if(len(moves) == 0):
              ####print("Fine")
              return "Stop"
            if moves[0] not in actions:
              ###print("Azione illegale")
              return "Stop"
            return moves[0]      

          if(len(moves) == 0):
            ####print("Fine")
            return "Stop"
          if moves[0] not in actions:
            ###print("Azione illegale")
            return "Stop"
          ####print("I should go to the best food")
          myState.wasPacman = myState.isPacman
          return moves[0]
    else: #I'm not pacman
          self.target =  None
          max_score = 0
          for i in range(0, self.layout.width):
            for j in range(0, self.layout.height):
              if(self.score_map[i][j]>max_score and self.walls[i][j]==False):
                self.target = (i,j)
                max_score = self.score_map[i][j]
          ####print("Il mio target è", self.target)
          path = astar(self.walls, gameState.getAgentState(self.index).getPosition(), self.target , self, False , self.has_eaten_capsule)
          #self.debugDraw([self.target], [1,1,1])      

          moves = path_to_moves(path)
          if(len(moves) == 0):
            ####print("Fine")
            return "Stop"
          if moves[0] not in actions:
            ###print("Azione illegale")
            return "Stop"
          ####print("I should go to the best food")
          myState.wasPacman = myState.isPacman
          return moves[0]
