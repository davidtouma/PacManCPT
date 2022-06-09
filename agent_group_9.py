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

from operator import truediv
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import pathCalculator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveDummyAgent', second = 'DefensiveDummyAgent'):
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

class DummyAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  # Used to share gameStates between friends
  primary_gameState = None
  secondary_gameState = None

  def registerInitialState(self, gameState):
    """
    
    """
    # run captureAgent.registerInitialState fist to prevent wonky results
    CaptureAgent.registerInitialState(self, gameState)
    
    # AGENT information
    self.current_agent_state = gameState.getAgentState(self.index)
    self.start = gameState.getAgentPosition(self.index)
    self.current_pos = (int(gameState.getAgentState(self.index).getPosition()[0]),\
      int(gameState.getAgentState(self.index).getPosition()[1]))

    # MAP information
    walls = gameState.getWalls()
    self.wall_map = walls.data
    self.map_width = walls.width
    self.map_height = walls.height

    # TEAM information
    self.team_ids = self.getTeam(gameState)
    self.primary = self.index == min(self.team_ids)
    for id in self.team_ids:
      if id != self.index:
        self.other_index = id
    
    # Share game state with ally
    self.other_gameState = None 
    if self.primary:
      DummyAgent.primary_gameState = gameState
    else:
      DummyAgent.secondary_gameState = gameState

    self.other_gameState = None 
    if self.primary and DummyAgent.secondary_gameState is not None:
      self.other_gameState = DummyAgent.secondary_gameState
    elif not self.primary and DummyAgent.primary_gameState is not None:
      self.other_gameState = DummyAgent.primary_gameState
   
    # OPPONENT stuff
    self.opponent_ids = self.getOpponents(gameState)
    self.opponent_starts = [gameState.getInitialAgentPosition(i) for i in self.opponent_ids]
    self.known_time = [0 for i in self.opponent_ids]

    # Define probability 
    self.prob_maps = [[[False for j in range(self.map_height)] for i in range(self.map_width)] for n in range(len(self.opponent_ids))]
    for n in range(len(self.opponent_ids)):
      self.prob_maps[n][self.opponent_starts[n][0]][self.opponent_starts[n][1]] = True

    # Initialize path planner
    self.pather = pathCalculator.Pather(gameState.data.layout)
   
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    self.getAndShare(gameState)

    self.makeObservation(gameState)

    self.doMapping(gameState)

    return self.pickAction(gameState)

  #####################################################################################################
  # Helper functions
  #####################################################################################################

  def getAndShare(self, gameState):
    ### Share gameState with friend  
    # Update shared game state
    if self.primary:
      DummyAgent.primary_gameState = gameState
    else:
      DummyAgent.secondary_gameState = gameState

    # update friend's gameState
    if self.primary and DummyAgent.secondary_gameState is not None:
      self.other_gameState = DummyAgent.secondary_gameState
    elif not self.primary and DummyAgent.primary_gameState is not None:
      self.other_gameState = DummyAgent.primary_gameState

    # At the start, other gamestates are a little wonky, only use other_gameState if valid_other is true
    # I think it not a problem after the initial turns
    if len(self.other_gameState.agentDistances) == 4:
      self.other_valid = True
    else:
      self.other_valid = False
  
    ### Record keeping
    ## Basic Info
    # valid actions
    self.actions = gameState.getLegalActions(self.index)
    self.current_agent_state = gameState.getAgentState(self.index)
    
    # positions of current agent and ally agent
    self.current_pos = (int(self.current_agent_state.getPosition()[0]),\
      int(self.current_agent_state.getPosition()[1]))

    self.other_pos = self.other_gameState.getAgentState(self.other_index).getPosition()
    
    # previous gameState
    self.prev_gameState = self.getPreviousObservation()

    ## store relevent information about each opponent in lists
    # Record the positions of opponent agents, will be None if out of range
    self.positions = []
    for n in self.opponent_ids:
      self.positions.append(gameState.getAgentPosition(n))

    self.other_positions = []
    for n in self.opponent_ids:
      if self.other_valid:
        self.other_positions.append(self.other_gameState.getAgentPosition(n))
      else:
        self.other_positions.append(None)

    # Record distanced of opponant agents, these are noisy +/- 6 (uniform) noise
    self.distances = []
    for n in self.opponent_ids:
      self.distances.append(gameState.getAgentDistances()[n])

    self.other_distances = []
    for n in self.opponent_ids:
      if self.other_valid:
        self.other_distances.append(self.other_gameState.getAgentDistances()[n])
      else:
        self.other_positions.append(-1)
    
    # Record the opponent agent states, useful for checking if an opponent is a pacman
    self.states = []
    for n in self.opponent_ids:
      self.states.append(gameState.getAgentState(n))

    # Record the opponent agent states, useful for checking if an opponent is a pacman
    self.prev_states = []
    for n in self.opponent_ids:
      if self.prev_gameState is None:
        self.prev_states.append(None)
        continue
      self.prev_states.append(self.prev_gameState.getAgentState(n))

  def makeObservation(self, gameState):
    ### Transitions I, Basic transitions that use previous state
    ## Food
    # Use the previous state to determine if interesting transitions have happened
    if self.red:
      self.currentFood = gameState.getRedFood()
      if self.prev_gameState is not None:
        oldFood = self.prev_gameState.getRedFood()
    else:
      self.currentFood = gameState.getBlueFood()
      if self.prev_gameState is not None:
        oldFood = self.prev_gameState.getBlueFood()

    # check for food being eaten
    self.food_taken_map = [[False for j in range(self.map_height)] for i in range(self.map_width)]

    if self.prev_gameState is not None:
      for i in range(self.map_width):
        for j in range(self.map_height):
          if oldFood[i][j] == True and self.currentFood[i][j] == False:
            self.food_taken_map[i][j] = True

    ## Pacman
    # two types of tranisition
    # ghost -> pacman: these are always valid localize opponent to x = map_width/2 - 1 (value: 1)
    # pacman -> ghost: these can represent two cases (value: -1)
    # 1) moving to home side
    # 2) getting eaten!, not sure how to tell apart could used the measurement data but its very noisy
    # TODO: ignore (2) for now, 
    self.pacman_transition = []
    for n in range(len(self.opponent_ids)):
      if self.prev_states[n] is None:
        self.pacman_transition.append(0)
        continue
      if self.prev_states[n].isPacman == False and self.states[n].isPacman == True:
        self.pacman_transition.append(1)
      elif self.prev_states[n].isPacman == True and self.states[n].isPacman == False:
        self.pacman_transition.append(-1)
      else:
        self.pacman_transition.append(0)

  def doMapping(self, gameState):

    ### Mapping

    ## Measurements
    # Create maps for each opponant that are consistant with the following conditions
    # - Walls
    # - Food locations
    # - Side of the map from the isPacman agent state
    # - Noisy distanse measurements
    
    location_maps = self.make_location_maps(self.current_pos,self.positions, self.distances, self.currentFood, self.states)

    ## build a map consistant friend's measurements
    # currently does not include any state info
    if self.other_valid:
      other_location_maps = self.make_location_maps(self.other_pos,self.other_positions,self.other_distances,None,None)
    else:
      #
      other_location_maps = [[[True for j in range(self.map_height)] for i in range(self.map_width)] for n in range(len(self.opponent_ids))]

    # Due to the age of your friends measurements, they are dilated
    other_location_maps = self.dilateMaps(other_location_maps)

    ### Transitions II (maybe unify w/ Transitions I)(depends on location_maps)
    ## Handles opponents getting eaten
    # transition from pacman -> ghost & measurements consistant with respawning
    for n in range(len(self.opponent_ids)):
      if self.pacman_transition[n] == -1 or location_maps[n][self.opponent_starts[n][0]][self.opponent_starts[n][1]] == True:
        # TODO: should I reset the whole prob_map <- probably not as the measurements will cull wayward 'particles'
        i_respawn = self.opponent_starts[n][0]
        j_respawn = self.opponent_starts[n][1]
        
        self.prob_maps[n][i_respawn][j_respawn] = True
        # Apply some dilation for safety, better to be more spread out than lose tracking
        # Dilate along x, i, and y, j
        for dilate in [-1,1]:
          if self.wall_map[i_respawn + dilate][j_respawn] == False:
            self.prob_maps[n][i_respawn + dilate][j_respawn] = True
      
          if self.wall_map[i_respawn][j_respawn + dilate] == False:
            self.prob_maps[n][i_respawn][j_respawn + dilate] = True

    ## probabilistic map
    # create map of all possible locations based on last known location(s)
    # two conditions can reset the probabilistic map
    # 1) a positive position reading
    # 2) a positive eating event that can be associated with only one <= maybe this is too strict
    # 2) Cont'd need to think about how to use food eating info... (TODO: should also add the capsule)

    for n in range(len(self.opponent_ids)):
      # Direct observation of this opponent 
      if self.positions[n] is not None:
        # Reset prob_map age
        self.known_time[n] = 0
        # Reset the prob_map
        temp_prob_map = [[False for j in range(self.map_height)] for i in range(self.map_width)]
        temp_prob_map[self.positions[n][0]][self.positions[n][1]] = True
        self.prob_maps[n] = temp_prob_map
        continue
      
      # No Direct measurement
      # Do not reset the prob map of i, dilate to model motion
      # increment age of prob_map, might be useful for determining when to ignore or reset the prob_map
      self.known_time[n] += 1
      temp_prob_map = self.dilateProbMap(n)
      
      # REMOVE Temp check
      # temp_check = [[False for j_temp in range(self.map_height)] for i_temp in range(self.map_width)]
      # for i_check in range(self.map_width):
      #   for j_check in range(self.map_height):
      #     if temp_prob_map[i_check][j_check] == True and self.prob_maps[n][i_check][j_check] == False:
      #       temp_check[i_check][j_check] = True


      # Update the prob_map
      self.prob_maps[n] = temp_prob_map
      
      # A ghost -> pacman transition was detected for this opponent
      if self.pacman_transition[n] == 1:
    
        if self.red:
          transition_index = self.map_width//2 - 1
        else:
          transition_index = self.map_width//2 

        # form the new prob_map
        temp_prob_map = [[False for j in range(self.map_height)] for i in range(self.map_width)]
        for j in range(self.map_height):
          # not wall and consistant with current measurement
          if self.wall_map[transition_index][j] == False and location_maps[n][transition_index][j] == True and self.prob_maps[n][transition_index][j]:# -FIX-
            temp_prob_map[transition_index][j] = True

        # Compare the prob_maps and use the best one
        current_prob_score = sum(x.count(True) for x in self.prob_maps[n])
        temp_prob_score = sum(x.count(True) for x in temp_prob_map)

        if temp_prob_score < current_prob_score:
          self.known_time[n] = 0
          self.prob_maps[n] = temp_prob_map

    ## Update probabilistic map I, taking into account the measurements
    # The opponents can not be in areas inconsistant with the measurements
    # opponents can be in regions not covered by the prob_maps if they are eaten and that isnt detected
    for n in range(len(self.opponent_ids)):
      for i in range(self.map_width):
        for j in range(self.map_height):
          ### FIX ### dont use old data when transition is detected
          if self.pacman_transition[n] == -1:
            # Ignore other data when a transition back to ghosthood is detected
            if location_maps[n][i][j] == False:
              self.prob_maps[n][i][j] = False
          else:
            if location_maps[n][i][j] == False or other_location_maps[n][i][j] == False:
              self.prob_maps[n][i][j] = False

    ## update probabilistic map II, taking into account food being eaten
    # TODO:it could be useful to repeat this if the first eaten food is covered by both opponents and the second(food) in only covered by one(opponent) for example
    for i in range(self.map_width):
        for j in range(self.map_height):
          if self.food_taken_map[i][j] == True:
            # check how many agents could be eating the food
            overlap_count = 0
            selected_agent = -1
            for n in range(len(self.opponent_ids)):
              if self.prob_maps[n][i][j] == True:
                overlap_count += 1
                selected_agent = n

            # count_sl = [self.prob_maps[n][i][j] for n in range(len(self.opponent_ids))].count(True)
            if overlap_count == 1:
              temp_prob_map = [[False for j_temp in range(self.map_height)] for i_temp in range(self.map_width)]
              temp_prob_map[i][j] = True
              self.prob_maps[selected_agent] = temp_prob_map

    # 'Unpack' the measurement map into lists for plotting
    # draw_measures[0] is list of 'first' opponent, [1] or 'second' opponent, [2] of both
    draw_measures = self.measurementMap2List(location_maps)

    # just plot one opponent for clarity
    # TODO: add all opponents
    draw_probs_0 = self.map2List(self.prob_maps[0])
    draw_probs_1 = self.map2List(self.prob_maps[1])
    
    draw_path = self.pather.getPath(self.current_pos, (15,7), self.positions[0], self.positions[1])

    # Remove
    min_ind = min(self.team_ids)
    current_ind = self.index
    test_1 = current_ind == min_ind
    #if self.red and self.index == min(self.team_ids):
      #self.debugClear()
      #self.debugDraw(draw_measures[0], [1,0,0], True)
      #self.debugDraw(draw_measures[1], [0,0,1], False)
      # self.debugDraw(draw_measures[2], [1,0,1], False)
      # if draw_path:
      # self.debugDraw(draw_path, [1,.5,1], False)
      #self.debugDraw(draw_probs_0, [0,0,1], True)
      #self.debugDraw(draw_probs_1, [0,.5,1], False)

  def make_location_maps(self, my_pos, positions, distances, currentFood, states):
      
    # define empty maps
    location_maps = [[[False for j in range(self.map_height)] for i in range(self.map_width)] for n in range(len(self.opponent_ids))]

    for n in range(len(self.opponent_ids)):
      # for each opponent
      if positions[n] is not None:
        location_maps[n][positions[n][0]][positions[n][1]] = True

      else:
        for i in range(self.map_width):
          for j in range(self.map_height):
            # check for walls
            if self.wall_map[i][j] == True:
              continue

            # check for food
            if currentFood is not None and currentFood.data[i][j] == True:
              continue

            # check for pacman state
            if states is not None:
              if self.red:
                if states[n].isPacman == True and i >= self.map_width/2:
                  continue
                if states[n].isPacman == False and i < self.map_width/2:
                  continue
              else:
                if states[n].isPacman == True and i < self.map_width/2:
                  continue
                if states[n].isPacman == False and i >= self.map_width/2:
                  continue

            # If none of the above and within the bounds of the noisy measurement its a valid location
            man_dist = abs(my_pos[0] - i) + abs(my_pos[1] - j)
            dist_meas = distances[n]
            if max(6, dist_meas - 6) <= man_dist <= (dist_meas + 6):
              location_maps[n][i][j] = True
    
    return location_maps

  def dilateProbMap(self, n):
    """
    Dilate the nth probability map
    """

    temp_prob_map = [[False for j_temp in range(self.map_height)] for i_temp in range(self.map_width)]
    for i in range(1, self.map_width - 1):
      for j in range(1, self.map_height - 1):
        if self.wall_map[i][j] == True:
          continue
        if self.prob_maps[n][i][j] == True:
          temp_prob_map[i][j] = True
          continue
        # check the directions on motion 
        if self.prob_maps[n][i-1][j] == True:
          temp_prob_map[i][j] = True
          continue
        if self.prob_maps[n][i+1][j] == True:
          temp_prob_map[i][j] = True
          continue
        if self.prob_maps[n][i][j-1] == True:
          temp_prob_map[i][j] = True
          continue
        if self.prob_maps[n][i][j+1] == True:
          temp_prob_map[i][j] = True
          continue
    
    return temp_prob_map

  def dilateMaps(self, maps):
    """
    Dilate the nth probability map
    """
    map_count = len(maps)

    temp_maps = [[[False for j in range(self.map_height)] for i in range(self.map_width)] for n in range(map_count)]
    
    for n in range(map_count):
      for i in range(1, self.map_width - 1):
        for j in range(1, self.map_height - 1):
          if self.wall_map[i][j] == True:
            continue
          if maps[n][i][j] == True:
            temp_maps[n][i][j] = True
            continue
          # check the directions on motion 
          if maps[n][i-1][j] == True:
            temp_maps[n][i][j] = True
            continue
          if maps[n][i+1][j] == True:
            temp_maps[n][i][j] = True
            continue
          if maps[n][i][j-1] == True:
            temp_maps[n][i][j] = True
            continue
          if maps[n][i][j+1] == True:
            temp_maps[n][i][j] = True
            continue
    
    return temp_maps

  def map2List(self, map):
    """
    Convert a map, list of lists, to list of tuples to plot
    """
    width = len(map)
    if width > 0:
      height = len(map[0])

    #if width != self.map_width or height != self.map_height:
      #print("Problem: map size mis-match!!")
    
    output_list = []
    for i in range(width):
        for j in range(height):
          if map[i][j] == True or map[i][j] == 1:
            output_list.append((i,j))

    return output_list

  def measurementMap2List(self, location_maps):
    """
    Returns a list of tuples corresponding to the opponents
    [0] first opponent only 
    [1] second opponent only
    [2] both opponents
    """
    draw_measures = [[],[],[]]
    for i in range(self.map_width):
          for j in range(self.map_height):
            if location_maps[0][i][j] == True and location_maps[1][i][j] == False:
              draw_measures[0].append((i,j))
            if location_maps[0][i][j] == False and location_maps[1][i][j] == True:
              draw_measures[1].append((i,j))
            if location_maps[0][i][j] == True and location_maps[1][i][j] == True:
              draw_measures[2].append((i,j))

    return draw_measures
  
  def safePathHome(self, positions):
    """
    Find shortest 'safe' path back to home side

    if on home side return remain
    """
    # Find valid destination
    if self.red:
      home_index = self.map_width//2 - 1
    else:
      home_index = self.map_width//2
    destinations = []
    for j in range(self.map_height):
      if self.wall_map[home_index][j] == False:
        destinations.append((int(home_index),int(j)))

    # Find paths home
    paths = []
    lengths = []
    for path_ind in range(len(destinations)):
      # Find and record path
      new_path = self.pather.getPath(self.current_pos, destinations[path_ind], positions[0], positions[1])
      paths.append(new_path)

      if new_path is not None:
        lengths.append(len(new_path))
      else:
        # This is basically just a large number so that invalid paths are ignored later
        lengths.append(self.map_width * self.map_height)

    # Pick the 'best' path
    # also checking for situation in which no valid paths are found
    if min(lengths) == self.map_width * self.map_height:
      return 'Stuck', [self.current_pos]
    best_path_ind = lengths.index(min(lengths))

    # Determine next action given best path
    start_coord = paths[best_path_ind][0]
    second_coord = paths[best_path_ind][1]

    delta_i = second_coord[0] - start_coord[0]
    delta_j = second_coord[1] - start_coord[1]

    # Check for malformed path?
    #if abs(delta_i) + abs(delta_j) > 1:
      #print("Problem with path, delta too large")

    # Default return action is 'Stop'
    # If the path indicates 
    return_action = 'Stop'

    if delta_i == 1:
      # EAST
      if 'East' in self.actions:
        return_action = 'East'
    elif delta_i == -1:
      # WEST
      if 'West' in self.actions:
        return_action = 'West'
    elif delta_j == 1:
      # NORTH
      if 'North' in self.actions:
        return_action = 'North'
    elif delta_j == -1:
      # SOUTH
      if 'South' in self.actions:
        return_action = 'South'

    return return_action, paths[best_path_ind]

  def getActionFromPositions(self, position1, position2):

    action = 'Stop'
    delta_i = position2[0] - position1[0]
    delta_j = position2[1] - position1[1]

    if delta_i == 1:
      # EAST
      if 'East' in self.actions:
        action = 'East'
    elif delta_i == -1:
      # WEST
      if 'West' in self.actions:
        action = 'West'
    elif delta_j == 1:
      # NORTH
      if 'North' in self.actions:
        action = 'North'
    elif delta_j == -1:
      # SOUTH
      if 'South' in self.actions:
        action = 'South'

    return action
  
  def getNewPositionfrmAction(self, action):
    current_i = int(self.current_pos[0])
    current_j = int(self.current_pos[1])

    if action == 'East':
      return (current_i + 1, current_j)
    elif action == 'West':
      return (current_i - 1, current_j)
    elif action == 'North':
      return (current_i, current_j + 1)
    elif action == 'South':
      return (current_i, current_j - 1)
    else:
      return (current_i, current_j)
  
  #####################################################################################################
  # Provided
  #####################################################################################################

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

  def pickAction(self, gameState):
    return random.choice(gameState.getLegalActions(self.index))

  #####################################################################################################
  # Agents
  #####################################################################################################

class OffensiveDummyAgent(DummyAgent):
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

    caps = self.getCapsules(successor)
    if len(caps) > 0: # This should always be True,  but better safe than sorry
      capCount = 0
      for cap in caps:
        capCount += 1
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, cap) for cap in caps])
      features['distanceToCap'] = minDistance
      features['capCount'] = capCount
    else:
      features['distanceToCap'] = 0
      features['capCount'] = 0


    return features

  def getWeights(self, gameState, action):
    if self.states[0] is not None and self.states[0].scaredTimer > 0:
      return {'successorScore': 100, 'distanceToFood': -1, 'distanceToCap': 0, 'capCount': 1000}
    # Default weights
    return {'successorScore': 100, 'distanceToFood': -1, 'distanceToCap': -2.5, 'capCount': -100}
    
  def pickAction(self, gameState):
    #####################################################################################################
    # High Level decision making
    #####################################################################################################
    # Parameters
    flee_distance = 5
    flee_food_count = 7 
    attack_dist = 3 # If any opponent pac man is detected within this range, ATTACK!
    max_dead_end = 4

    ### if near enemy pacman attack!
    for n in range(len(self.opponent_ids)):
      if self.positions[n] is not None and\
        self.current_agent_state.scaredTimer == 0 and\
        (self.states[n].isPacman == True or\
        (self.states[n].scaredTimer > 4 and self.states[n].scaredTimer <= 40)):
        victim_dist = self.distancer.getDistance(self.current_pos, self.positions[n])
        if victim_dist <= attack_dist:
          attack_path = self.pather.getPath(self.current_pos, self.positions[n])
          if len(attack_path) > 1:
            move = self.getActionFromPositions(attack_path[0], attack_path[1])
            #print("Giving chase!!")
            self.FSM_state = "Attack"
            return move
          #else:
            #print("was going to give chase but had a problem!")

    ### continue diverting
    if self.FSM_state == 'Diverting':
      if self.current_pos[0] >= self.home_index or self.current_pos == self.goal:
        self.FSM_state = "Normal"
        self.goal = (-1, -1)
      else:
        blocked_locations = [(self.home_index + 1, j) for j in range(self.map_height)]
        blocked_locations.remove(self.goal)
        divert_path = self.pather.getPath_withoutlist(self.current_pos, self.goal, blocked_locations)
        move = self.getActionFromPositions(divert_path[0], divert_path[1])
        if move in self.actions:
          #print("Diverting - {}".format(len(divert_path)))
          #if self.red and self.index == min(self.team_ids):
            #self.debugDraw(divert_path, [0,1,0], False)
          return move
        #else:
          #print("problem following diversion!")

    ### Don't jump into sharks mouth
    # if on boundry and opponents arent scared
    positions_count = 0
    for element in self.positions:
      if element is not None:
        positions_count += 1

    if self.current_pos[0] == self.home_index and\
        positions_count > 0 and\
        self.states[self.opponent_ids[0]].scaredTimer == 0 and\
        self.wall_map[self.current_pos[0] + 1][self.current_pos[1]] == False: 
      ## maybe a function
      """
      record valid cross over points and check them for safety
      """
      possible_destinations = []
      possible_scores = []
      for j in range(self.map_height):
        if self.wall_map[self.home_index + 1][j] == False and self.wall_map[self.home_index][j] == False:
          valid_pos = (self.home_index + 1, j)
          possible_destinations.append(valid_pos)
          scores = []
          for n in range(len(self.opponent_ids)):
            if self.positions[n] is not None:
              dist = self.distancer.getDistance(self.positions[n], valid_pos)
              scores.append(dist)
          possible_scores.append(min(scores))

      # Now I need to figure out what to do with these scores
      # Only consider destinations that are safe, the known enemies travel dist exceeds some threshold
      possible_paths = []
      possible_path_scores = []
      for p in range(len(possible_destinations)):
        if possible_scores[p] > flee_distance:
          # Find path on our side
          blocked_locations = [(self.home_index + 1, j) for j in range(self.map_height)]
          blocked_locations.remove(possible_destinations[p])
          possible_path = self.pather.getPath_withoutlist(self.current_pos, possible_destinations[p], blocked_locations)
          
          if possible_path is not None and len(possible_path) > 1:
            possible_paths.append(possible_path)
            possible_path_scores.append(len(possible_path))

      # hopefully have a safe path somewhere
      #if len(possible_paths) == 0:
        #print("problem finding a safe alternative path")

      else:
        if len(possible_paths) > 1:
          best_path = possible_paths[random.randint(0,len(possible_paths)-1)]
          #best_path = possible_paths[possible_path_scores.index(min(possible_path_scores))]
          self.FSM_state = "Diverting"
          self.goal = best_path[len(best_path) - 1]

          #print("Found route to cross safely - {}".format(len(best_path)))
          # if self.index == min(self.team_ids):
          #     self.debugDraw(best_path, [0,1,0], True)
          move = self.getActionFromPositions(best_path[0], best_path[1])
          return move
        
    ### Avoid attackers, 
    # valid Attackers
    attacker_count = 0
    attacker_distances = []
    attacker_positions = []
    for n in range(len(self.opponent_ids)):
      # only consider nearby and 'defensive' opponents
      # avoid scared agent with timers that are almost up
      if self.current_agent_state.isPacman and \
        self.positions[n] is not None and \
        self.states[n].isPacman == False and \
        self.states[n].scaredTimer <= 4:
        attacker_dist = self.distancer.getDistance(self.current_pos, self.positions[n])
        # NEW
        attacker_count += 1
        attacker_distances.append(attacker_dist)
        attacker_positions.append(self.positions[n])
    if self.current_agent_state.isPacman:
      if attacker_count > 0: # or self.current_agent_state.numCarrying >= flee_food_count:
        safe_move, safe_path = self.safePathHome(self.positions)
        if min(attacker_distances) <= flee_distance:
          # Enter retreat state
          self.FSM_state = "Retreat"
          #print("Retreating 1")
          
          if safe_move != 'Stuck':
            # Debug plotting
            # if self.index == min(self.team_ids):
            #   self.debugDraw(safe_path, [1,.5,0], True)
            return safe_move
          
          #else:
            # Doing nothing her corresponds to just eating until the problem resolves itself
            #print("Run: I'm stuck!!")
            #return 'Stop'
        else:
          # Check path
          if safe_move != 'Stuck':
            spare_moves, conflict = self.check_early_retreat(safe_path, attacker_positions,max_dead_end)
            if spare_moves <= 2 and spare_moves >=0 :
              self.FSM_state = "Retreat-Early"
              #print("Retreating 2")
              return safe_move

    ### Don't be greedy
    if self.current_agent_state.isPacman and \
      self.states[n].scaredTimer <= 4 and \
      self.current_agent_state.numCarrying >= flee_food_count:
      self.FSM_state = "Returning"
      move, safe_path = self.safePathHome(self.positions)
        
      if move != 'Stuck':
        # Debug plotting
        #if self.red and self.index == min(self.team_ids):
          #self.debugDraw(safe_path, [1,.5,0], False)
        return move
      
      #else:
        #print("Greedy: I'm stuck!!")


    #####################################################################################################
    # normal food gathering behavior
    #####################################################################################################
    self.FSM_state = "Gathering"
    generic_action = self.genericFoodGathering(gameState)

    return generic_action
  
  def registerInitialState(self, gameState):
    DummyAgent.registerInitialState(self, gameState)

    self.FSM_state = 'Start'
    self.goal = (-1, -1)

    if self.red:
      self.home_index = self.map_width//2 - 1
    else:
      self.home_index = self.map_width//2

  def check_early_retreat(self, safe_path, attacker_positions, max_depth):
    """
    This is our way of checking if 
    """
    spare_moves = 0
    min_spare_moves = 10000
    my_travel_dist = 0
    attacker_travel_dist = 0
    intersection = (-1,-1)

    # check for weirdness
    if attacker_positions is None or safe_path is None:
      return -3, intersection
    if len(safe_path) < max_depth:
      return -2, intersection
    if len(attacker_positions) == 0:
      return -1, intersection

    for n in range(len(attacker_positions)):
      # Find the attacking path of each attacker, hate attackes...
      attack_path = self.pather.getPath(attacker_positions[n], self.current_pos)
      for safe_path_ind in range(1, max_depth):
        my_travel_dist = safe_path_ind
        attacker_travel_dist = self.distancer.getDistance(attacker_positions[n], safe_path[safe_path_ind])

        if attacker_travel_dist <= my_travel_dist:
          spare_moves = 0
        else:
          spare_moves = attacker_travel_dist - my_travel_dist

        if spare_moves < min_spare_moves:
          min_spare_moves = spare_moves
          intersection = safe_path[safe_path_ind]
        min_spare_moves = min(min_spare_moves, spare_moves)

    return spare_moves, intersection
  
  def genericFoodGathering(self, gameState):
    values = [self.evaluate(gameState, a) for a in self.actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(self.actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in self.actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

class DefensiveDummyAgent(DummyAgent):
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

  def pickAction(self, gameState):

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    # [index][x][y]
    # [n][i][j]
    # self.prob_maps

    #####################################################################################################
    # High Level decision making
    #####################################################################################################

    # Finding out capsules
    capsules = self.getCapsulesYouAreDefending(gameState)

    # Checking scaredness
    scared = False
    myself = gameState.getAgentState(self.index)
    if (myself.scaredTimer > 0 and myself.scaredTimer <= 40):
      scared = True

    # Finding boundary
    layout = gameState.data.layout
    allNodes = layout.walls.asList(False)
    boundaryNodes = []
    guardNodes = []
    if self.red:
      boundaryIndex = self.map_width//2 - 1
      guardIndex = self.map_width//2 - 2
    else:
      boundaryIndex = self.map_width//2
      guardIndex = self.map_width//2 + 1
    
    for node in allNodes:
      (i, j) = node
      if i == boundaryIndex:
        boundaryNodes.append((i,j))
      elif i == guardIndex:
        guardNodes.append((i,j))

    # Resetting chase logic
    chase = False

    # (PINK) Looking for enemies nearby on our side
    if not chase:
      for n in range(len(self.opponent_ids)):
        if self.red:
          if self.positions[n] is not None and self.positions[n][0] <= boundaryIndex:
            chasee = self.positions[n]
            chase = True
            color = [1,.5,1]
        else:
          if self.positions[n] is not None and self.positions[n][0] >= boundaryIndex:
            chasee = self.positions[n]
            chase = True
            color = [1,.5,1]

    # (YELLOW) Looking for someone eating food
    if not chase:
      for i in range(self.map_width):
        for j in range(self.map_height):
          if self.food_taken_map[i][j]:
            chasee = (i,j)
            chase = True
            color = [1,1,.5]

    # (MINT) Looking for enemies nearby anywhere
    if not chase:
      for n in range(len(self.opponent_ids)):
        if self.red:
          if self.positions[n] is not None:
            chasee = self.positions[n]
            chase = True
            color = [.5,1,.5]
      
      # Fix target to our side of the boundary
      if chase:
        if self.red:
          if chasee[0] > boundaryIndex:
            dist = 10000
            for node in guardNodes:
              newDist = self.getMazeDistance(node, chasee)
              if newDist < dist:
                closest = node 
                dist = newDist
            chasee = closest
        else:
          if chasee[0] < boundaryIndex:
            dist = 10000
            for node in guardNodes:
              newDist = self.getMazeDistance(node, chasee)
              if newDist < dist:
                closest = node
                dist = newDist
            chasee = closest

    # (CYAN) Move to most likely position of closest enemy
    if not chase:
      color = [.5,1,1]
      dist = 10000
      for n in range(len(self.opponent_ids)):
        for i in range(self.map_width):
          for j in range(self.map_height):
            if self.prob_maps[n][i][j]:
              newDist = self.getMazeDistance(gameState.getAgentPosition(self.index),(i,j))
              if newDist < dist:
                chase = True
                chasee = (i,j)
                dist = newDist
      if chase:
        # Finding a guard point if there are capsules
        if capsules:
          dist = 10000
          for node in allNodes:
            distToClosest = self.getMazeDistance(node, chasee)
            distToFirstCapsule = self.getMazeDistance(node, capsules[0])
            distToSecondCapsule = self.getMazeDistance(node, capsules[-1])
            distances = [distToClosest, distToFirstCapsule, distToSecondCapsule]
            newDist = max(distances)
            if newDist < dist:
              best = node
              dist = newDist
          chasee = best
        else:
          # Otherwise matches the vertical position of the closest enemy
          if self.red:
            if chasee[0] > boundaryIndex:
              dist = 10000
              for node in guardNodes:
                newDist = self.getMazeDistance(node, chasee)
                if newDist < dist:
                  closest = node 
                  dist = newDist
              chasee = closest
          else:
            if chasee[0] < boundaryIndex:
              dist = 10000
              for node in guardNodes:
                newDist = self.getMazeDistance(node, chasee)
                if newDist < dist:
                  closest = node
                  dist = newDist
              chasee = closest

    # Scared behavior means staying between 2 and 3 cells away from enemies
    if chase and scared:
      perimiter = []
      for node in allNodes:
        if self.getMazeDistance(node, chasee) > 1 and \
          self.getMazeDistance(node, chasee) <= 3:
          perimiter.append(node)
      dist = 10000
      for node in perimiter:
        newDist = self.getMazeDistance(gameState.getAgentPosition(self.index), node)
        if newDist < dist:
          dist = newDist
          closest = node
      chasee = closest
      color = [1, 0.75, 0.5]
               
    #####################################################################################################
    # Low level planning
    #####################################################################################################

    # Plan path to target and choose action along path
    if chase:
      if gameState.getAgentPosition(self.index) == chasee:
        return 'Stop'
      else:
        path = self.pather.getPath(gameState.getAgentPosition(self.index), chasee)
        #self.debugDraw(path, color, True)
        action = self.getActionFromPositions(gameState.getAgentPosition(self.index), path[1])
        return action

    # values = [self.evaluate(gameState, a) for a in self.actions]

    # maxValue = max(values)
    # bestActions = [a for a, v in zip(self.actions, values) if v == maxValue]

    # foodLeft = len(self.getFood(gameState).asList())

    # if foodLeft <= 2:
    #   bestDist = 9999
    #   for action in self.actions:
    #     successor = self.getSuccessor(gameState, action)
    #     pos2 = successor.getAgentPosition(self.index)
    #     dist = self.getMazeDistance(self.start,pos2)
    #     if dist < bestDist:
    #       bestAction = action
    #       bestDist = dist
    #   return bestAction

    return random.choice(self.actions)

