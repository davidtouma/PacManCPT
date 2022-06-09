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


from turtle import position
from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions
import game
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import copy
from capture import halfGrid
from game import Grid
#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='SimpleAgent', second='SimpleAgent'):
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

class SimpleAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """
    testVar = 5
    initialized = False
    isRed = []
    positions = []
    def setupBelief(self,gameState):
        wall = gameState.getWalls()
        width = wall.width
        height = wall.height
        if not self.__class__.initialized:
            self.__class__.isRed = [True, True, True, True]
            self.__class__.positions = [np.zeros((width,height),int), np.zeros((width,height),int), np.zeros((width,height),int), np.zeros((width,height),int)]
            self.__class__.spawnPositions = [(0,0),(0,0),(0,0),(0,0)]
            self.__class__.initialized = True
        idx = self.index
        self.__class__.isRed[idx] = False
        position = gameState.getAgentPosition(self.index)
        enemyIdx = (idx-1)%4
        self.__class__.spawnPositions[enemyIdx] = (width-position[0]-1,height-position[1]-1)
        self.__class__.positions[enemyIdx][width-position[0]-1,height-position[1]-1] = 1
        #print("ENEMY BELIIIFEI")
        #print(self.__class__.positions[enemyIdx])
    def updatePositions(self,gameState):
        idx = self.index
        enemyIdx = (idx-1)%4
        self.killEnemyAgents(gameState)
        self.__class__.positions[enemyIdx] = self.stepOponent(gameState,enemyIdx)
        self.updatePositionsFromDistance(gameState)
        self.updatePositionsFromDirectObs(gameState)
        
        #print(self.__class__.positions[enemyIdx])
    def manhattanDist(self,p1,p2):
        return int(abs(p1[0]-p2[0])+np.abs(p1[1]-p2[1]))
    def killEnemyAgents(self,gameState):
        current_observation = self.getCurrentObservation()
        previous_observation = self.getPreviousObservation()
        if current_observation == None or previous_observation == None:
            return
        ownPosCurrent = current_observation.getAgentPosition(self.index)
        ownPosPrevious = previous_observation.getAgentPosition(self.index)
        #If we moved more than one step we were the ones who died
        wall = gameState.getWalls()
        if self.manhattanDist(ownPosCurrent,ownPosPrevious) > 2:
            return
        for idx in self.getOpponents(gameState):
            
            posCurrent = current_observation.getAgentPosition(idx)
            posPrevious = previous_observation.getAgentPosition(idx)
            #If the enemy position was known before, is unknown now, and previous distance was less than 2
            if posCurrent == None and posPrevious == None:
                continue
            if not posCurrent == None and not posPrevious == None:
                continue
            if not posCurrent == None:
                continue
            if self.manhattanDist(ownPosCurrent,posPrevious) == 0:
                #enemy died
                self.__class__.spawnPositions[idx] = np.zeros((wall.width,wall.height),int)   
                enemyspawn_i = self.__class__.spawnPositions[idx][0]
                enemyspawn_j = self.__class__.spawnPositions[idx][1]
                self.__class__.positions[idx][enemyspawn_i,enemyspawn_j] = 1
                
    def updatePositionsFromDistance(self,gameState):
        wall = gameState.getWalls()
        current_observation = self.getCurrentObservation()
        x = np.linspace(0, wall.width-1, wall.width).astype(int)
        y = np.linspace(0, wall.height-1, wall.height).astype(int)
        xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
        position = gameState.getAgentPosition(self.index)
        xdist = xgrid - position[0]
        ydist = ygrid - position[1]
        
        manhatan = abs(xdist) + abs(ydist)
        
        for idx in self.getOpponents(gameState):
            reading = current_observation.getAgentDistances()[idx]
            #Enemy is within 7 units of the reading
            posiblePositions1 = (abs(manhatan-reading) < 7).astype(int)
            #Enemy is also at least 6 units away else we would have seen them
            posiblePositions2 = (manhatan > 5).astype(int)
            posiblePositions = np.multiply(posiblePositions1,posiblePositions2)
            #print(posiblePositions)
            self.__class__.positions[idx] = np.multiply(self.__class__.positions[idx], posiblePositions)
    def updatePositionsFromDirectObs(self,gameState):
        for idx in self.getOpponents(gameState):
            pos = self.getCurrentObservation().getAgentPosition(idx)
            if not pos == None:
                self.__class__.positions[idx] = np.zeros(
                    self.__class__.positions[idx].shape, int)
                self.__class__.positions[idx][pos[0], pos[1]] = 1.
    
    def stepOponent(self,gameState,idx):
        wall = gameState.getWalls()
        start = self.__class__.positions[idx]
        steps = self.makeShortestPathGrid(wall,start)
        spread = (steps < 2).astype(int)
        return spread
    def makHomeTurfGrid(self,grid):
        halfway = int(grid.width / 2)
        self.hometurf = Grid(grid.width, grid.height, False)
        if self.red:    xrange = range(halfway)
        else:       xrange = range(halfway, grid.width)

        for y in range(grid.height):
            for x in xrange:
                self.hometurf[x][y] = True
    def _debug_msg(self):
        print("simpleAgent")
    def pathGridFromAgentPosition(self,gameState):
        position = gameState.getAgentPosition(self.index)
        return self.PathGridFromPoint(gameState,position)
    
    def pathGridFromHome(self, gameState):
        wall = gameState.getWalls()
        npWall = np.array(wall.data,int)
        #print("WAT")
        #print(npWall)
        grid = np.ones((wall.width, wall.height),int)
        grid-=npWall
        half = int(wall.width/2)
        #print(half)
        if self.red:
            grid[half:,:] = 0
        else:
            grid[0:half,:] = 0
        #print(grid)
        return self.makeShortestPathGrid(wall,grid)
    def PathGridFromPoint(self,gameState,position):
        wall = gameState.getWalls()
        start = np.zeros((wall.width, wall.height),int)
        start[position[0],position[1]] = 1
        return self.makeShortestPathGrid(wall,start)
    
    def pathGridFromEnemies(self,gameState):
        wall = gameState.getWalls()
        belief = np.zeros((wall.width, wall.height),int)
        for idx in self.getOpponents(gameState):
            belief += self.__class__.positions[idx] > 0
            
        belief = (belief > 0).astype(int)
        self.plotGrid(belief-1,(-0.1,0.1))
        return self.makeShortestPathGrid(wall,belief)
    
    def pathGridFromFood(self,gameState):
        wall = gameState.getWalls()
        food = np.array(self.getFood(gameState).data,int)
        return self.makeShortestPathGrid(wall,food)
    def registerInitialState(self, gameState):
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
        print('Initialising simple agent')
        print(self.index)
        print(len(gameState.data.agentStates))
        self.start = gameState.getAgentPosition(
            self.index)  # copied from baseline
        CaptureAgent.registerInitialState(self, gameState)
        self.step = 0
        #print(self.start)

        # can we do better for the initial belief than uniform?
        # just enemy half, .....

        '''
    Your initialization code goes here, if you need any.
    '''
        self.setupBelief(gameState)
        wall = gameState.getWalls()
        #print("here are walls")
        #print(wall)
        self.enemy_p_belief = {}
        # self.enemy_mht_belief={}
        self.insibilitySteps = 0
        own_pos = gameState.getAgentPosition(self.index)
        self.numCarrying = 0
        print('startup oppinets')
        print(self.getOpponents(gameState))
        self.makHomeTurfGrid(self.getFood(gameState))
        
        #print(self.hometurf)
        for idx in self.getOpponents(gameState):
            self.enemy_p_belief[idx] = np.zeros(
                (wall.width, wall.height), float)
            # self.enemy_mht_belief[idx]=np.zeros((wall.width+wall.height-1),float)
            _pos = gameState.getInitialAgentPosition(idx)
            self.enemy_p_belief[idx][_pos[0], _pos[1]] = 1.

            # mht_dist=np.abs(own_pos[0]-_pos[0])+np.abs(own_pos[1]-_pos[1])
            # self.enemy_mht_belief[idx][mht_dist]=1.

        # TODO: bfs from the entries to enemy/own territory to flee ASAP?
        self.distancer.getMazeDistances()  # thus we can access them fast
    def plotGrid(self,grid,range):
        upper = range[1]
        lower = range[0]
        points = list(zip(*np.where(np.multiply(grid > lower,grid < upper))))
        #print(points)
        self.debugDraw(points, [0.5, 0.5, 0.], True)
    def chooseAction(self, gameState):
        if self.step == 0:
            self.setupBelief(gameState)
        self.step += 1
        self.updatePositions(gameState)
        
        self.numCarrying = gameState.data.agentStates[self.index].numCarrying
        own_team = self.getTeam(gameState)
        #print(self.__class__.testVar)
        if self.__class__.testVar == 5 and self.index == 1:
            self.__class__.testVar = 10
        current_observation = self.getCurrentObservation()
        previous_observation = self.getPreviousObservation()
        own_pos = gameState.getAgentPosition(self.index)

        enemy_idx = self.getOpponents(gameState)

        ### update beliefs ################################

        #self.belief_transition(
            #current_observation, previous_observation, self.uniform_policy)

        #if self.index == min(own_team):

            #for x in range(self.enemy_p_belief[min(enemy_idx)].shape[0]):
                #for y in range(self.enemy_p_belief[min(enemy_idx)].shape[1]):

                    # _man=np.abs(own_pos[0]-x)+np.abs(own_pos[1]-y)

                    #if x == 0 and y == 0:
                        #delete = True
                    #else:
                        #delete = False
                    #self.debugDraw([(x,y)],[0.,min(1e12*self.enemy_p_belief[min(enemy_idx)][x,y],1.),0.],delete)

                    #self.debugDraw([(x,y)],[0.,min(1.,20*self.enemy_p_belief[min(enemy_idx)][x,y]),0.],delete)

                    # self.debugDraw([(x,y)],[0.,self.p_reading_given_pos(reading,(x,y),gameState.getAgentPosition(self.index))*12.,0.],False)

            # _=self.get_rl_state_from_positions_flee_attacker(own_pos,[(5,5),(5,5)],current_observation,4,True)
            # print(_)

        ######################################################
        #self.closest_escape(gameState)
        
        #actions = [self.find_closest_enemy_food(gameState)]
        #dir, length = self.find_closest_enemy_food(gameState)
        dir, length = self.find_closest_enemy_food(gameState)
        #print(str(self.getFood(gameState).width) + ', ' + str(self.getFood(gameState).height))
        self.enemyGridPaths = self.pathGridFromEnemies(gameState)
        self.foodGridPaths = self.pathGridFromFood(gameState)
        self.agentPaths = self.pathGridFromAgentPosition(gameState)
        self.homeDistancePaths = self.pathGridFromHome(gameState)
        if(self.is_on_enemy_territory(gameState)):
            if self.is_invinsible(gameState):
                actions = self.find_closest_enemy_food_invinsible(gameState)
            else:
                if(self.chase_capsule(gameState)):
                    # loot action placeholder
                    actions, _ = self.find_closest_capsule(gameState)
                else:
                    if(self.loot(gameState)):
                        # chase enemy power capsule placeholder
                        actions = self.find_closest_enemy_food2(gameState)
                    else:
                        # flee placeholder
                        actions, _ = self.closest_escape(gameState)
        else:
            if(self.defend_own_territory(gameState)):
                if(self.chase_agents(gameState)):
                    # chase enemy pacmans placeholder
                    actions = gameState.getLegalActions(self.index)
                else:
                    # take defensive positions placeholder
                    actions = gameState.getLegalActions(self.index)
            else:
                # head to enemy territory placeholder
                actions = self.find_closest_enemy_food2(gameState)

        """
    Picks among actions randomly.
    """
        # actions = gameState.getLegalActions(self.index)

        '''
        
    You should change this in your own agent.
    '''
        #print(actions)
        # return 'east' 'west' 'south' or 'north'
        return random.choice(actions)

    def is_on_enemy_territory(self, gameState):

        state = gameState.getAgentState(self.index)
        return state.isPacman

    def loot(self, gameState):
        '''
        function to decide wheter we should go for enemy food
        (called on enemy terrain)
        '''
        enemy_food = self.getFood(gameState)
        numberOfFoods = np.sum(np.sum(np.array(enemy_food.data),axis=1),axis=0)
        if numberOfFoods < 3:
            return False
        _ , length = self.find_closest_enemy_food(gameState)
        return  (self.numCarrying == 0) or (self.numCarrying == 1 and length < 5)  or (self.numCarrying == 2 and length < 4)  or (self.numCarrying == 3 and length < 3)  or (self.numCarrying == 4 and length < 2)or (self.numCarrying == 5 and length < 1)
        return False

    def closest_enemy(self):
        pass

    def closest_escape(self,gameState):
        """
        find the quickest path to escape back to own territory
        should return 'stop' 0 and if starting on friendly terrority
        """
        pass
        pos = gameState.getAgentPosition(self.index)
        wall = gameState.getWalls()
        dir, path = self.bfs_until(pos, lambda x: self.hometurf[x[0]][x[1]], wall, False)
        length = len(path)
        if len(path) > 1:
            dir = self.convert_neighbour_to_action(pos, path[-2])
        elif len(path) == 1:
            dir =  self.convert_neighbour_to_action(pos, path[-1])
        else:
            dir = 'Stop'
        return [dir], length
    def bfs_until(self, position, is_goal_territory, wall, plot=False):
        '''
        function to perform BFS from given position until it gets to the goal territory (eg. enemy food, home terrain,...) 
        takes current agent position, a matrix of target squares, walls and optional plot argument.
        returns direction and path
        make sure the goal terrain aint empty!
        '''

        def generate_valid_bfs_neighbours(position, wall):

            valid_neighbours = []

            for dv in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                new_pos = (position[0]+dv[0], position[1]+dv[1])

                # out of the terrain
                if new_pos[0] < 0 or new_pos[0] >= wall.width or new_pos[1] < 0 or new_pos[1] > wall.height:
                    continue

                if not wall[new_pos[0]][new_pos[1]]:
                    valid_neighbours.append(new_pos)

            return valid_neighbours

        pos_queue = deque()
        pos_queue.append(position)
        covered = set([position])
        parent = {position: (-1, -1)}

        # we don't check for non-existing goal, the game shall be finished by then by the central coordinatior

        aim = None

        while(len(pos_queue) > 0):
            current = pos_queue.popleft()  # queue would be better

            neigbours = generate_valid_bfs_neighbours(current, wall)

            for n in neigbours:
                if not (n in covered):
                    pos_queue.append(n)
                    covered.add(n)
                    parent[n] = current

                    if is_goal_territory(n):
                        aim = n
                        pos_queue = []
                        break

        path = []
        current = aim

        if aim:
            path.append(current)
            _parent = parent[current]

            while _parent[0] > -1:
                path.append(_parent)
                _parent = parent[_parent]

        if plot:
            #print(path)
            self.debugDraw(path, [0.5, 0.5, 0.], True)
        #print("path length is " + str(len(path)))
        return aim, path
    def get_optimal_path(self, gameState):
        p = gameState.getAgentPosition(self.index)
        foods = np.array(self.getFood(gameState).data,int)
        F = [self.agentPaths]
        for food in foods:
            F.append(self.PathGridFromPoint(gameState, food))
        A = self.agentPaths
        H = self.homeDistancePaths
        self.optirec(H,F,p,0,self.insibilitySteps,[])
        
    def optirec(self,H,F,p,i,steps,path):
        if H[position[0],position[1]] > steps:
            return []
        else:
            longest = 0
            for j in range(0,len(H)):
                steps = steps-F[i][p[0]][p[1]]
                path = self.optirec(H,F,p,j,steps,path)
                return []
            
        
         
    def is_invinsible(self,gameState):
        try:
            current_capsules = self.getCapsules(self.getCurrentObservation())
            previous_capsules = self.getCapsules(self.getPreviousObservation())
        except: 
            return False
        position = gameState.getAgentPosition(self.index)
        if self.insibilitySteps > 0:
            self.insibilitySteps -= 1
            print("agent " + str(self.index) + " is invisible! " + str(self.insibilitySteps))
        if not len(current_capsules) == len(previous_capsules):
            self.insibilitySteps = 40
            print("agent " + str(self.index) + " became invisible!")
        return self.insibilitySteps > 0
        
    def find_closest_enemy_food_invinsible(self,gameState):
        print("WOOO LETS GOOOO")
        position = gameState.getAgentPosition(self.index)
        wall = gameState.getWalls()
        i = position[0]
        j = position[1]
        if self.insibilitySteps - 1 < self.homeDistancePaths[i,j]:
            moveGrid = self.homeDistancePaths
        else:
            moveGrid = self.foodGridPaths
        return self.get_best_moves_on_grid(moveGrid,position,wall)
    
    def get_best_moves_on_grid(self, moveGrid,position,wall):
        cross = [(1,0),(-1,0),(0,1),(0,-1)]
        i = position[0]
        j = position[1]
        bestMove = 999999
        bestDir = Directions.STOP
        for offset in cross:
            ii = offset[0]
            jj = offset[1]
            if not wall[ii+i][jj+j] and moveGrid[ii+i][jj+j] < bestMove:
                bestMove = moveGrid[ii+i][jj+j]
                bestDir = self.convert_neighbour_to_action(position,(ii+i,jj+j))
                #print(bestDir)
        return [bestDir]
    def find_closest_enemy_food2(self,gameState):
        position = gameState.getAgentPosition(self.index)
        i = position[0]
        j = position[1]
        closestEnemy = self.enemyGridPaths[i,j]
        escapeDistance = 4
        
        wall = gameState.getWalls()
        alpha = (1,-1)[self.is_home_territory(gameState,position,wall) and closestEnemy < 4]
        gamma = (closestEnemy/escapeDistance*0.5+0.5,0)[alpha == -1]
        moveGrid = alpha*(-self.enemyGridPaths+self.agentPaths)+gamma*self.foodGridPaths
        moveGrid -= moveGrid[i,j]
        return self.get_best_moves_on_grid(moveGrid,position,wall)

    def find_closest_enemy_food(self, gameState):
        '''
        BFS to closest enemy food, return a move towards
        '''

        wall = gameState.getWalls()
        enemy_food = self.getFood(gameState)
        pos = gameState.getAgentPosition(self.index)

        aim, path = self.bfs_until(
            pos, lambda x: enemy_food[x[0]][x[1]], wall, False)

        if len(path) > 1:
            dir = self.convert_neighbour_to_action(pos, path[-2])
        elif len(path) == 1:
            dir =  self.convert_neighbour_to_action(pos, path[-1])
        else:
            dir = 'Stop'
        length = len(path)
        return [dir], length
    def find_closest_capsule(self, gameState):
        '''
        BFS to closest capsule, return a move towards
        '''

        wall = gameState.getWalls()
        enemy_capsule = self.getCapsules(gameState)
        pos = gameState.getAgentPosition(self.index)

        if len(enemy_capsule) == 0:
            return gameState.getLegalActions(self.index), 0

        aim, path = self.bfs_until(
            pos, lambda x: x in enemy_capsule, wall, True)

        if len(path) > 1:
            dir = self.convert_neighbour_to_action(pos, path[-2])
        elif len(path) == 1:
            dir =  self.convert_neighbour_to_action(pos, path[-1])
        else:
            dir = 'Stop'
        length = len(path)
        return [dir], length
    def is_home_territory(self, gameState, cell, wall):
        '''
        determine if a given cell is traversable home territory

        '''

        # wall=gameState.getWalls()
        halfway = wall.width/2

        # I think this is the right way to assess

        if self.red:
            return cell[0] < halfway and wall[cell[0]][cell[1]] == False
        else:
            return cell[0] >= halfway and wall[cell[0]][cell[1]] == False

    def flee_home(self, gameState):
        '''
        try to get home as fast a possible
        '''

        wall = gameState.getWalls()
        pos = gameState.getAgentPosition(self.index)

        aim, path = self.bfs_until(
            pos, lambda x: self.is_home_territory(gameState, x, wall), wall, True)

        if len(path) > 1:
            return self.convert_neighbour_to_action(pos, path[-2])
        else:
            return self.convert_neighbour_to_action(pos, path[-1])

    def chase_capsule(self, gameState):
        enemy_capsule = self.getCapsules(gameState)
        if not len(enemy_capsule) == 0:
            i = enemy_capsule[0][0]
            j = enemy_capsule[0][1]
            return self.enemyGridPaths[i,j]-self.agentPaths[i,j] > 0
        else:
            return False

    def defend_own_territory(self, gameState):
        '''
        decide if we shall head for enemy terrain to gain food or remain in our own to defend 
        (called on own terrain)
        '''

        return False

    def chase_agents(self, gameState):
        '''
        function to decide whether we chase enemy pacmans or protect territory
        (called on own terrain)
        '''

        return True

    def convert_neighbour_to_action(self, position, neighbour):
        '''
        to convert desired next gridcell to move (convenience method)

        we assume it is indeed a valid neighbour!
        '''

        dx = neighbour[0]-position[0]
        dy = neighbour[1]-position[1]

        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST

        if dy == 1:
            return Directions.NORTH
        else:
            return Directions.SOUTH

    def get_rl_state_from_positions_flee_attacker(self, own_pos, enemy_positions, current_observation, n_exits=4, debug_plot=False):
        '''
        generate state representation for the ML part, consisting of the distance matrix of own position, enemy positions and exits, and waypoints halfway to them

        positions assumed to be given as tuples (as usual),
        enemy_positions shall be a LIST of them

        '''

        points = [own_pos]+enemy_positions

        # get the exits - local minima for distances from us

        wall = current_observation.getWalls()
        halfway = int(wall.width/2)

        if self.red:
            x_exit = halfway-1
        else:
            x_exit = halfway

        exit_distances = []
        exit_points = []

        for y in range(wall.height):
            if not wall[x_exit][y]:
                exit_distances.append(
                    self.getMazeDistance(own_pos, (x_exit, y)))
                exit_points.append((y, exit_distances[-1]))
            else:
                exit_distances.append(10*wall.height*wall.width)

        local_minima = []

        if exit_distances[0] < exit_distances[1]:
            local_minima.append((0, exit_distances[0]))
        if exit_distances[-1] < exit_distances[-2]:
            local_minima.append((wall.height-1, exit_distances[-1]))

        for i in range(1, wall.height-1):
            if exit_distances[i] < exit_distances[i+1] and exit_distances[i] < exit_distances[i-1] and exit_distances[i] < 10*wall.height*wall.width:
                local_minima.append((i, exit_distances[i]))

        local_minima.sort(key=lambda x: x[1])

        if len(local_minima) > n_exits:
            for i in range(n_exits):
                points.append((x_exit, local_minima[i][0]))
        else:
            points = points+[(x_exit, y[0]) for y in local_minima]

            # padding out the gap between n_exit and the actual number of exits
            # just add some random guys from the same column

            for y in np.random.permutation(exit_points):
                if not((x_exit, y[0]) in points) and len(points) < (3+n_exits):
                    points.append((x_exit, y[0]))

        if len(points) < 3+n_exits:
            while(len(points) < 3+n_exits):
                points.append(
                    current_observation.getInitialAgentPosition(self.index))

        # get halfway points to exits

        for i in range(3, 3+n_exits):
            points.append(self.generate_halfway(own_pos, points[i], wall))

        # calculate distances

        distances = []

        for i in range(len(points)-1):
            for j in range(i, len(points)):
                distances.append(self.getMazeDistance(points[i], points[j]))

        # return normalized values - not really [0,1], just to scale back from huge values to o(1)

        _dist = np.array(distances, float)

        if debug_plot:
            for p in range(len(points)):
                self.debugDraw(points[p], [0., 0.5, 0.5], p == 0)

        return _dist/(wall.height+wall.width)

    def npToGrid(array):
        pass
    def gridToNp(grid,type):
        pass
    def makeShortestPathGrid(self,walls,start):
        """Creates a grid of the map with the number of moves to get to every point in the map a given set of starting point(s)
            Walls is a grid object with true and false if there is a wall in a given spot. Start is a numpy array which is 1 where we start and 0 otherwise"""
        newPoints = deque(zip(*np.where(start > 0.5)))
        #print("ZIOP")
        #print(newPoints)
        pathMatrix = np.full((walls.width, walls.height),99999, float)
        pathMatrix[start == 1] = 0
        cross = [(1,0),(-1,0),(0,1),(0,-1)]
        while len(newPoints) > 0:
            point = newPoints.popleft()
            i = point[0]
            j = point[1]
            value = pathMatrix[i,j]
            for offset in cross:
                ii = offset[0]
                jj = offset[1]
                if not walls[ii+i][jj+j] and pathMatrix[ii+i][jj+j] > value + 1:
                    #print("Appended")
                    newPoints.append((ii+i,jj+j))
                    pathMatrix[ii+i][jj+j] = value + 1
        #print(pathMatrix) 
        return pathMatrix
    def get_rl_state_from_positions_flee_defender(self, own_pos, enemy_position, teammate_pos, current_observation, n_exits=4, debug_plot=False):
        '''
        generate state representation for the ML part, consisting of the distance matrix of own position, enemy positions and exits, and waypoints halfway to them

        positions assumed to be given as tuples (as usual),
        enemy_positions shall be a LIST of them

        '''

        points = [own_pos, teammate_pos, enemy_position]

        # get the exits - local minima for distances from us

        wall = current_observation.getWalls()
        halfway = int(wall.width/2)

        if self.red:
            x_exit = halfway
        else:
            x_exit = halfway-1

        exit_distances = []
        exit_points = []

        for y in range(wall.height):
            if not wall[x_exit][y]:
                exit_distances.append(
                    self.getMazeDistance(enemy_position, (x_exit, y)))
                exit_points.append((y, exit_distances[-1]))
            else:
                exit_distances.append(10*wall.height*wall.width)

        local_minima = []

        if exit_distances[0] < exit_distances[1]:
            local_minima.append((0, exit_distances[0]))
        if exit_distances[-1] < exit_distances[-2]:
            local_minima.append((wall.height-1, exit_distances[-1]))

        for i in range(1, wall.height-1):
            if exit_distances[i] < exit_distances[i+1] and exit_distances[i] < exit_distances[i-1] and exit_distances[i] < 10*wall.height*wall.width:
                local_minima.append((i, exit_distances[i]))

        local_minima.sort(key=lambda x: x[1])

        if len(local_minima) > n_exits:
            for i in range(n_exits):
                points.append((x_exit, local_minima[i][0]))
        else:
            points = points+[(x_exit, y[0]) for y in local_minima]

            # padding out the gap between n_exit and the actual number of exits
            # just add some random guys from the same column

            for y in np.random.permutation(exit_points.keys):
                if not((x_exit, y) in points) and len(points) < (3+n_exits):
                    points.append((x_exit, y))

        if len(points) < 3+n_exits:
            while(len(points) < 3+n_exits):
                points.append(current_observation.getInitialAgentPosition(
                    self.getOpponents()[0]))

        # get halfway points to exits

        for i in range(3, 3+n_exits):
            points.append(self.generate_halfway(
                enemy_position, points[i], wall))

        # calculate distances

        distances = []

        for i in range(len(points)-1):
            for j in range(i, len(points)):
                distances.append(self.getMazeDistance(points[i], points[j]))

        # return normalized values - not really [0,1], just to scale back from huge values to o(1)

        _dist = np.array(distances, float)

        if debug_plot:
            for p in range(len(points)):
                self.debugDraw(points[p], [0., 0.5, 0.5], p == 0)

        return _dist/(wall.height+wall.width)

    def generate_halfway(self, pos1, pos2, walls):

        _, path = self.bfs_until(pos1, lambda x: x == pos2, walls, False)

        return path[int(len(path)/2)]

    def belief_transition(self, current_observation, previous_observation, enemy_policy):
        '''
        wrapper to update beliefs about the enemies

        order:
        1) direct observation
        2) noisy observation
        3) disapperaing food

        '''

        enemy_indices = [idx for idx in self.enemy_p_belief.keys()]

        for idx in enemy_indices:
            if not self.belief_transition_direct_observation(current_observation, idx):
                self.belief_transition_noisy_observation(
                    current_observation, idx, enemy_policy)

        if previous_observation == None:
            pass
        else:
            self.belief_transition_disappearing_food(
                current_observation, previous_observation)

        # since we have literally no clue about the enemy policy, we set every nonnegative entry to normaized uniform value

        for idx in enemy_indices:
            self.enemy_p_belief[idx] = np.array(
                self.enemy_p_belief[idx] > 0., float)
            #print(np.array(self.enemy_p_belief[idx] > 0., bool))
        # for idx in enemy_indices:
        #   if not self.belief_transition_manhattan_direct(current_observation,idx):
        #     self.belief_transition_manhattan_noisy(current_observation,idx)

        # if previous_observation==None:
        #   pass
        # else:
        #   self.belief_transition_manhattan_disappearing_food(current_observation,previous_observation)

    def belief_transition_direct_observation(self, current_observation, idx):
        '''
        update beliefs upon direct observation of agent of index idx
        '''

        pos = current_observation.getAgentPosition(idx)

        if not pos == None:
            self.enemy_p_belief[idx] = np.zeros(
                self.enemy_p_belief[idx].shape, float)
            self.enemy_p_belief[idx][pos[0], pos[1]] = 1.
            return True
        else:

            # TODO: remove from 5 Manhattan distance

            return False

    def belief_transition_disappearing_food(self, current_observation, previous_observation):
        '''
        account for the disappearing food by simultaneous updates of both adversaries' whereabout beliefs

        even though it is not the mathematically correct way to do this, we assume independence before and AFTER the observation, where the latter is not exactly true
        '''

        # find position of disappearenca

        if self.red:
            food = current_observation.getRedFood()
            prev_food = previous_observation.getRedFood()
            capsule = current_observation.getRedCapsules()
            prev_capsule = previous_observation.getRedCapsules()
            enemy_indices = current_observation.getBlueTeamIndices()
            # enemy_indices=[1,3]
        else:
            food = current_observation.getBlueFood()
            prev_food = previous_observation.getBlueFood()
            capsule = current_observation.getBlueCapsules()
            prev_capsule = previous_observation.getBlueCapsules()
            enemy_indices = current_observation.getRedTeamIndices()
            # enemy_indices=[2,4]

        gaps = []

        other_idx = {enemy_indices[0]: enemy_indices[1],
                     enemy_indices[1]: enemy_indices[0]}

        for x in range(food.width):
            for y in range(food.height):

                _tmp_pos = (x, y)

                # print _tmp_pos

                if (not food[x][y]) and prev_food[x][y]:
                    gaps.append((x, y))
                if (_tmp_pos in prev_capsule) and (not (_tmp_pos in capsule)):
                    gaps.append((x, y))

        if(len(gaps) == 0):
            return False
        if(len(gaps) == 1):
            gap = gaps[0]

            p_hit = {}
            for idx in self.enemy_p_belief.keys():
                p_hit[idx] = self.enemy_p_belief[idx][gap[0], gap[1]]

            for idx in self.enemy_p_belief.keys():
                self.enemy_p_belief[idx] = self.enemy_p_belief[idx] * \
                    p_hit[other_idx[idx]]/(p_hit[other_idx[idx]]
                                           * (1.-p_hit[idx])+p_hit[idx])
                self.enemy_p_belief[idx][gap[0], gap[1]] = p_hit[idx] / \
                    (p_hit[other_idx[idx]]*(1.-p_hit[idx])+p_hit[idx])

        if(len(gaps) == 2):

            p_g1 = {}
            p_g2 = {}
            for idx in self.enemy_p_belief.keys():
                p_g1[idx] = self.enemy_p_belief[idx][gaps[0][0], gaps[0][1]]
                p_g2[idx] = self.enemy_p_belief[idx][gaps[1][0], gaps[1][1]]

            for idx in self.enemy_p_belief.keys():
                self.enemy_p_belief[idx] = np.zeros(
                    self.enemy_p_belief[idx].shape, float)
                self.enemy_p_belief[idx][gaps[0][0], gaps[0]
                                         [1]] = p_g1[idx]*p_g2[other_idx[idx]]
                self.enemy_p_belief[idx][gaps[1][0], gaps[1]
                                         [1]] = p_g2[idx]*p_g1[other_idx[idx]]
                self.enemy_p_belief[idx] /= np.sum(self.enemy_p_belief[idx])

    def belief_transition_noisy_observation(self, current_observation, idx, enemy_policy):
        '''
        perform transiton on the belief state about the enemy location, supposing that we already have the noisy (Manhattan) distance reading
        and we approximate the adversarial behaviour by some MARKOV policy

        policy expected to return conditional probability of given step at given positon
         '''

        # we have 1-to-1 correspondance between matrix elements and terrain cells

        p_new = np.zeros(self.enemy_p_belief[idx].shape, float)
        p_matrix = self.enemy_p_belief[idx]
        own_pos = current_observation.getAgentPosition(self.index)

        reading = current_observation.getAgentDistances()[idx]

        wall = current_observation.getWalls()

        for x in range(p_new.shape[0]):
            for y in range(p_new.shape[1]):

                # Markovian evolution: P((x,y)|observations) ~ P(reading|(x,y))* sum_neighbours P((x_n,y_n)| observations up to prevoius step)

                p_new[x, y] = 0.

                if wall[x][y]:
                    continue

                # note that we have zeros at wall

                for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                    # cell from the grid
                    if x+dx >= 0 and x+dx < p_new.shape[0] and y+dy >= 0 and y+dy < p_new.shape[1]:
                        p_new[x, y] += p_matrix[x+dx, y+dy] * \
                            enemy_policy((x+dx, y+dy), (-dx, -dy),
                                         current_observation)

                p_new[x, y] = self.p_reading_given_pos(
                    reading, (x, y), own_pos)*p_new[x, y]

        self.enemy_p_belief[idx] = p_new/np.sum(p_new)

    def p_reading_given_pos(self, reading_, pos, own_pos):
        '''
        calculate the conditional probalbility of the reading given hidden position and reader position
        '''

        d_manhattan = np.abs(pos[0]-own_pos[0])+np.abs(pos[1]-own_pos[1])
        if np.abs(d_manhattan-reading_) <= 6:
            return 1/13.
        else:
            return 0.

    def belief_transition_manhattan_direct(self, current_observation, idx):

        pos = current_observation.getAgentPosition(idx)

        if not pos == None:

            own_pos = current_observation.getAgentPosition(self.index)

            _man = np.abs(own_pos[0]-pos[0])+np.abs(own_pos[1]-pos[1])

            self.enemy_mht_belief[idx] = np.zeros(
                self.enemy_mht_belief[idx].shape, float)
            self.enemy_mht_belief[idx][_man] = 1.
            return True
        else:

            self.enemy_mht_belief[idx][:6] *= 0.
            self.enemy_mht_belief[idx] /= np.sum(self.enemy_mht_belief[idx])

            return False

    def belief_transition_manhattan_disappearing_food(self, current_observation, previous_observation):

        if self.red:
            food = current_observation.getRedFood()
            prev_food = previous_observation.getRedFood()
            capsule = current_observation.getRedCapsules()
            prev_capsule = previous_observation.getRedCapsules()
            enemy_indices = current_observation.getBlueTeamIndices()
        else:
            food = current_observation.getBlueFood()
            prev_food = previous_observation.getBlueFood()
            capsule = current_observation.getBlueCapsules()
            prev_capsule = previous_observation.getBlueCapsules()
            enemy_indices = current_observation.getRedTeamIndices()

        gaps = []

        own_pos = current_observation.getAgentPosition(self.index)

        other_idx = {enemy_indices[0]: enemy_indices[1],
                     enemy_indices[1]: enemy_indices[0]}

        for x in range(food.width):
            for y in range(food.height):

                _tmp_pos = (x, y)

                # print _tmp_pos

                if (not food[x][y]) and prev_food[x][y]:
                    gaps.append((x, y))
                if (_tmp_pos in prev_capsule) and (not (_tmp_pos in capsule)):
                    gaps.append((x, y))

        if(len(gaps) == 0):
            return False
        if(len(gaps) == 1):
            gap = gaps[0]

            d_gap = np.abs(own_pos[0]-gap[0]) + \
                np.abs(own_pos[1]-gap[1])  # manhattan distance

            p_hit = {}
            for idx in self.enemy_mht_belief.keys():
                p_hit[idx] = self.enemy_mht_belief[idx][d_gap]

            for idx in self.enemy_mht_belief.keys():
                self.enemy_mht_belief[idx] = self.enemy_mht_belief[idx] * \
                    p_hit[other_idx[idx]]/(p_hit[other_idx[idx]]
                                           * (1.-p_hit[idx])+p_hit[idx])
                self.enemy_mht_belief[idx][d_gap] = p_hit[idx] / \
                    (p_hit[other_idx[idx]]*(1.-p_hit[idx])+p_hit[idx])

        if(len(gaps) == 2):

            p_g1 = {}
            p_g2 = {}

            dg1 = np.abs(own_pos[0]-gaps[0][0]) + \
                np.abs(own_pos[1]-gaps[0][1])  # manhattan distance
            dg2 = np.abs(own_pos[0]-gaps[1][0]) + \
                np.abs(own_pos[1]-gaps[1][1])  # manhattan distance

            for idx in self.enemy_mht_belief.keys():
                p_g1[idx] = self.enemy_mht_belief[idx][gaps[0][0], gaps[0][1]]
                p_g2[idx] = self.enemy_mht_belief[idx][gaps[1][0], gaps[1][1]]

            for idx in self.enemy_mht_belief.keys():
                self.enemy_mht_belief[idx] = np.zeros(
                    self.enemy_mht_belief[idx].shape, float)
                self.enemy_mht_belief[idx][gaps[0][0], gaps[0]
                                           [1]] = p_g1[idx]*p_g2[other_idx[idx]]
                self.enemy_mht_belief[idx][gaps[1][0], gaps[1]
                                           [1]] = p_g2[idx]*p_g1[other_idx[idx]]
                self.enemy_mht_belief[idx] /= np.sum(
                    self.enemy_mht_belief[idx])

                #print(self.enemy_mht_belief[idx])

    def belief_transition_manhattan_noisy(self, current_observation, idx):

        #print(self.enemy_mht_belief[idx])

        new_belief = np.zeros(self.enemy_mht_belief[idx].shape, float)
        old_belief = self.enemy_mht_belief[idx]
        reading = current_observation.getAgentDistances()[idx]

        # idea: Markovian observations again, assume that distance varies uniformly during 1 round

        for d in range(new_belief.shape[0]):

           # firts assemble neighbours

            valid_neighbours = 0
            for neigh in [-2, -1, 0, 1, 2]:
                if d+neigh >= 0 and d+neigh < new_belief.shape[0]:
                    new_belief[d] += old_belief[d+neigh]
                    valid_neighbours += 1
            new_belief[d] /= valid_neighbours

            # then multiply by conditional probability of the observed value given cell

        if np.abs(reading-d) <= 6:
            new_belief[d] /= 13.
        else:
            new_belief[d] = 0.

        new_belief = new_belief/np.sum(new_belief)

        self.enemy_mht_belief[idx] = new_belief

        #print(self.enemy_mht_belief[idx])

    def uniform_policy(self, position, action, current_observation):

        wall = current_observation.getWalls()

        no_negighbours = 1  # STOP is always allowed
        goal_pos = (position[0]+action[0], position[1]+action[1])

        for dv in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            new_pos = (position[0]+dv[0], position[1]+dv[1])

            # out of the terrain
            if new_pos[0] < 0 or new_pos[0] >= wall.width or new_pos[1] < 0 or new_pos[1] >= wall.height:
                if dv == action:
                    return 0.
                continue

            if not wall[new_pos[0]][new_pos[1]]:
                no_negighbours += 1
        if wall[goal_pos[0]][goal_pos[1]]:
            return 0.
        else:
            return 1./no_negighbours

    def initialize_flee_defense_model(self, n_in):
        '''
        initialize NN to predict Q(a,s) values for the fleeing attacker

        idea: simple connected network of few layers, predict single real value (and then train with experience replay in approximate Q-learn way)
        '''
