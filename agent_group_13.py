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


from captureAgents import CaptureAgent
import random, time, util
from graphicsUtils import *
from game import Directions
import capture
import game
import numpy as np
import heapq
import enum


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='Agent', second='Agent'):
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
class Agent(CaptureAgent):
    """
    Our agent
    """
    width = 0
    height = 0
    grid = []
    enemyIndex = [0, 0]
    enemyPos = [set(), set()] #Positions are tuples
    enemyFood = [0, 0]
    friendIndex = [0, 0]
    friendState = [None, None]
    friendLocked = [False, False]
    supportChoke = [-1,-1]
    targetIndex = -1
    prevFood = None
    enemyCapsule = None
    powerUp = False
    scared = False
    homeSide = True
    death = False
    actions = ['East', 'North', 'West', 'South', 'Stop']

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
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

        # Find agent indices and initial enemy locations
        friends = self.getTeam(gameState)
        enemies = self.getOpponents(gameState)
        Agent.friendIndex[0] = friends[0]
        Agent.friendIndex[1] = friends[1]
        Agent.enemyIndex[0] = enemies[0]
        Agent.enemyIndex[1] = enemies[1]
        Agent.friendState[0] = self.States.Choose
        Agent.friendState[1] = self.States.Choose
        self.powerTimer = 0
        self.turnTimer = 300
        # Initialize grid and enemy locations
        if self.index == Agent.friendIndex[0]:
            self.teamIndex = 0
            # Generate grid
            walls = gameState.getWalls()
            Agent.width = walls.width
            Agent.height = walls.height
            for x in range(walls.width):
                column = []
                for y in range(walls.height):
                    column.append(self.Node(x, y, walls[x][y]))
                Agent.grid.append(column)
            Agent.enemyPos[0].add(gameState.getInitialAgentPosition(enemies[0]))
            Agent.enemyPos[1].add(gameState.getInitialAgentPosition(enemies[1]))
            Agent.prevFood = self.getFoodYouAreDefending(gameState)
            Agent.enemyCapsule = self.getCapsulesYouAreDefending(gameState)
            self.findChokes()
            self.initFood(gameState)
            self.initCapsule(gameState)
        else:
            self.teamIndex = 1

    def findChokes(self):
        """
        Identify choke points in grid
        Add pointers to related choke points for each Node
        """
        for x in range(int(self.width / 2)):
            for y in range(self.height):
                if Agent.grid[x][y].wall:
                    continue
                neighbors = self.findNeighbors(x, y)
                # Check for one width choke points
                if len(neighbors) == 2:
                    neighborOne = neighbors.pop()
                    neighborOneNb = self.findNeighbors(neighborOne[0], neighborOne[1])
                    neighborTwo = neighbors.pop()
                    neighborTwoNb = self.findNeighbors(neighborTwo[0], neighborTwo[1])
                    corner = abs(neighborOne[0] - neighborTwo[0]) > 0 and abs(neighborOne[1] - neighborTwo[1]) > 0
                    # Place choke so it covers point with most available directions
                    # Check if corridor
                    if len(neighborOneNb) < 3 and len(neighborTwoNb) < 3:
                        continue
                    # Check if "open" corner
                    # Check if one side open, skip if closed side and I are corners
                    if len(neighborOneNb) < 3 and len(neighborTwoNb) > 2 and neighborTwo[0] < self.width / 2:
                        if self.isCorner(neighborOne) and corner:
                            continue
                        Agent.grid[neighborTwo[0]][neighborTwo[1]].choke = True
                        continue
                    elif len(neighborTwoNb) < 3 and len(neighborOneNb) > 2 and neighborOne[0] < self.width / 2:
                        if self.isCorner(neighborTwo) and corner:
                            continue
                        Agent.grid[neighborOne[0]][neighborOne[1]].choke = True
                        continue
                    # If both sides open and I am corner, I am not a choke
                    else:
                        if corner:
                            continue
                    # Set choke points between both open sides
                    Agent.grid[x][y].choke = True
        # Search for set of adjacent choke points, remove all members with < 2 neighbors
        for x in range(int(self.width / 2)):
            for y in range(self.height):
                if Agent.grid[x][y].choke:
                    chokes = {(x, y)}
                    chokes = self.findNeighborChokes(x, y, chokes)
                    if len(chokes) > 1:
                        for pos in chokes:
                            if len(self.findNeighbors(pos[0], pos[1])) < 3:
                                Agent.grid[pos[0]][pos[1]].choke = False
        # Add pointers to choke points that are related to each position
        globalVisited = set()  # set of explored pos
        segments = []  # list of explored segments
        chokesList = []  # list of sets of chokes corresponding to segments
        deadEnds = set()  # set of all dead end choke points
        for x in range(int(self.width / 2)):
            for y in range(self.height):
                if Agent.grid[x][y].wall or Agent.grid[x][y].choke:
                    continue
                if (x, y) in globalVisited:
                    continue
                visited, chokes = self.findLocalChokes((x, y), set(), set())
                globalVisited.union(visited)
                # Add choke points to set of all chokes and to set of all dead ends if it's a dead-end
                segments.append(visited)
                chokesList.append(chokes)
                if len(chokes) == 1:
                    deadEnds.union(chokes)
                for pos in visited:
                    Agent.grid[pos[0]][pos[1]].exits = chokes
        # Add connected border as exits
        for i in range(len(segments)):
            borders = set()
            for pos in segments[i]:
                if pos[0] == int(Agent.width / 2) - 1 and not Agent.grid[int(Agent.width / 2)][pos[1]].wall:
                    borders.add((pos[0], pos[1]))
            for pos in chokesList[i]:
                if pos[0] == int(Agent.width / 2) - 1 and not Agent.grid[int(Agent.width / 2)][pos[1]].wall:
                    borders.add((pos[0], pos[1]))
            if len(borders) > 0:
                for pos in segments[i]:
                    Agent.grid[pos[0]][pos[1]].exits.update(borders)
                for pos in chokesList[i]:
                    Agent.grid[pos[0]][pos[1]].exits.update(borders)
            for pos in borders:
                Agent.grid[pos[0]][pos[1]].scoreDist = 1
                Agent.grid[pos[0]][pos[1]].scorePos = (pos[0]+1, pos[1])
        for i in range(len(segments)):
            if len(chokesList[i]) > 1:
                nonExits = chokesList[i].intersection(deadEnds)
                for nonExit in nonExits:
                    # If choke point is dead end and only connected to one non-dead end choke point,
                    # then it's not an exit for non dead-end choke point
                    remove = True
                    for j in range(len(segments)):
                        if not remove:
                            break
                        if j == i:
                            continue
                        if nonExit in chokesList[j]:
                            for choke in chokesList[j]:
                                if choke not in deadEnds:
                                    remove = False
                    if remove:
                        for pos in segments[i]:
                            Agent.grid[pos[0]][pos[1]].exits.remove(nonExit)
        # Choke points share exits with all neighboring segments
        # Choke points know shortest distance to own side
        for x in range(int(Agent.width / 2)):
            for y in range(Agent.height):
                if Agent.grid[x][y].choke:
                    neighbors = self.findNeighborNodes(Agent.grid[x][y])
                    for neighbor in neighbors:
                        Agent.grid[x][y].exits.update(neighbor.exits)
                    bestDist = 1000000
                    bestPos = None
                    for h in range(Agent.height):
                        _, dist = self.findPath((x, y), (int(Agent.width / 2), h))
                        if dist < bestDist:
                            bestDist = dist
                            bestPos = (int(Agent.width / 2), h)
                    Agent.grid[x][y].scoreDist = bestDist
                    Agent.grid[x][y].scorePos = bestPos


        # Mirror chokes and exits to other side
        for x in range(int(Agent.width / 2)):
            for y in range(Agent.height):
                mirror = Agent.grid[Agent.width - x - 1][Agent.height - y - 1]
                mirror.choke = Agent.grid[x][y].choke
                mirrorExits = set()
                for exit in Agent.grid[x][y].exits:
                    mirrorExits.add((Agent.width - exit[0] -1, Agent.height - exit[1] - 1))
                mirror.exits = mirrorExits
                scorePos = Agent.grid[x][y].scorePos
                if x == int(Agent.width / 2) - 1:
                    if scorePos is not None:
                        mirror.scoreDist = Agent.grid[x][y].scoreDist
                        mirror.scorePos = (Agent.width - scorePos[0] - 1, Agent.height - scorePos[1] - 1)
                if mirror.choke:
                    mirror.scoreDist = Agent.grid[x][y].scoreDist
                    mirror.scorePos = (Agent.width - scorePos[0] - 1, Agent.height - scorePos[1] - 1)



    def isCorner(self, pos):
        """
        Check if pos is a corner
        """
        neighbors = self.findNeighbors(pos[0], pos[1])
        firstX = None
        firstY = None
        if len(neighbors) == 2:
            for neighbor in neighbors:
                if firstX == None:
                    firstX = neighbor[0]
                    firstY = neighbor[1]
                    continue
                if abs(neighbor[0] - firstX) > 0 and abs(neighbor[1] - firstY) > 0:
                    return True
        return False

    def findLocalChokes(self, pos, visited, chokes):
        """
        Find segment of nodes and their choke points
        """
        visited.add(pos)
        neighbors = self.findNeighbors(pos[0], pos[1]) - visited
        for neighbor in neighbors:
            if Agent.grid[neighbor[0]][neighbor[1]].choke:
                chokes.add(neighbor)
                continue
            visited, chokes = self.findLocalChokes(neighbor, visited, chokes)
        return visited, chokes

    def findNeighborChokes(self, x, y, chokes):
        """
        Find neighboring chokes of grid at location x,y
        """
        offsets = [-1, 1]
        neighbors = set()
        for offset in offsets:
            newX = x + offset
            newY = y + offset
            if -1 < newX < self.width:
                if Agent.grid[newX][y].choke:
                    neighbors.add((newX, y))
            if -1 < newY < self.height:
                if Agent.grid[x][newY].choke:
                    neighbors.add((x, newY))
        if len(neighbors) == 0:
            return chokes
        for choke in neighbors:
            if choke not in chokes:
                chokes.add(choke)
                chokes = self.findNeighborChokes(choke[0], choke[1], chokes)
        return chokes

    def findNeighbors(self, x, y):
        """
        Find neighbors of grid at location x,y
        """
        offsets = [-1, 1]
        neighbors = set()
        for offset in offsets:
            newX = x + offset
            newY = y + offset
            if -1 < newX < self.width:
                if not Agent.grid[newX][y].wall:
                    neighbors.add((newX, y))
            if -1 < newY < self.height:
                if not Agent.grid[x][newY].wall:
                    neighbors.add((x, newY))
        return neighbors


    def updateEnemyPositions(self, gameState):
        """
        Updates possible locations for enemies, also updates their food
        """
        distances = gameState.getAgentDistances()
        newFood = self.getFoodYouAreDefending(gameState)
        newCapsule = self.getCapsulesYouAreDefending(gameState)
        # Sort enemies depending on who moved last
        movedEnemyIndex = self.index-1
        if movedEnemyIndex < 0:
            movedEnemyIndex = 3
        if movedEnemyIndex == Agent.enemyIndex[0]:
            enemyIndices = [Agent.enemyIndex[0], Agent.enemyIndex[1]]
            enemyPos = [Agent.enemyPos[0], Agent.enemyPos[1]]
            localEnemyIndex = [0, 1]
        else:
            enemyIndices = [Agent.enemyIndex[1], Agent.enemyIndex[0]]
            enemyPos = [Agent.enemyPos[1], Agent.enemyPos[0]]
            localEnemyIndex = [1, 0]
        # Update agents possible locations
        # Add new possible locations to moved agent
        addPos = set()
        for pos in enemyPos[0]:
            offsets = [-1, 1]
            for offset in offsets:
                newX = pos[0] + offset
                newY = pos[1] + offset
                if -1 < newX < self.width:
                    if not gameState.hasWall(newX, pos[1]):
                        addPos.add((newX, pos[1]))
                if -1 < newY < self.height:
                    if not gameState.hasWall(pos[0], newY):
                        addPos.add((pos[0], newY))
        for pos in addPos:
            enemyPos[0].add(pos)
        # Check if agents are directly observable
        for i in range(2):
            if gameState.getAgentPosition(enemyIndices[i]) is not None:
                enemyPos[i].clear()
                enemyPos[i].add(gameState.getAgentPosition(enemyIndices[i]))
        # Prune based on food pickup
        for x in range(self.width):
            for y in range(self.height):
                if self.prevFood[x][y] and not newFood[x][y]:
                    enemyPos[0].clear()
                    enemyPos[0].add((x, y))
                    Agent.enemyFood[localEnemyIndex[0]] += 1
        # Prune based on capsule pickup
        if len(Agent.enemyCapsule) > len(newCapsule):
            for eaten in Agent.enemyCapsule:
                if eaten not in newCapsule:
                    enemyPos[0].clear()
                    enemyPos[0].add(eaten)
                    break
            Agent.enemyCapsule = newCapsule
        # Prune based on state and distances
        for i in range(2):
            # Update food count if ghost
            if not gameState.getAgentState(enemyIndices[i]).isPacman:
                Agent.enemyFood[localEnemyIndex[0]] = 0
            if len(enemyPos[i]) == 1:
                continue
            removePos = set()
            pruneRed = True
            if (gameState.getAgentState(enemyIndices[i]).isPacman and self.red) or (not gameState.getAgentState(enemyIndices[i]).isPacman and not self.red):
                pruneRed = False
            for pos in enemyPos[i]:
                # Prune based on state
                if pruneRed:
                    if pos[0] < self.width/2:
                        removePos.add(pos)
                        continue
                else:
                    if pos[0] > self.width/2 - 1:
                        removePos.add(pos)
                        continue
                # Prune based on observed dist and proximity
                distPos = self.getManhattanDistance(gameState.getAgentPosition(self.index), pos)
                if distPos < distances[enemyIndices[i]] - 6 or distPos > distances[enemyIndices[i]] + 6 or distPos < 6:
                    removePos.add(pos)
            for pos in removePos:
                enemyPos[i].remove(pos)
        Agent.prevFood = newFood
        # Update grid with probable enemy locations, add extra grid spots in order to avoid known enemy locations
        for x in range(self.width):
            for y in range(self.height):
                Agent.grid[x][y].enemy = False
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i in range(2):
            posLen = len(enemyPos[i])
            # Failsafe add spawn point if no points
            if posLen == 0:
                enemyPos[i].add(gameState.getInitialAgentPosition(Agent.enemyIndex[i]))
            for pos in enemyPos[i]:
                Agent.grid[pos[0]][pos[1]].enemy = True
                if posLen == 1:
                    for i, dir in enumerate(directions):
                        if Agent.grid[pos[0] + dir[0]][pos[1] + dir[1]].wall:
                            continue
                        nextDirs = [(0, 0), directions[i - 1 if i > -1 else 3], dir, directions[i + 1 if i < 3 else 0]]
                        for nDir in nextDirs:
                            if -1 < pos[0] + dir[0] + nDir[0] < Agent.width and -1 < pos[1] + dir[1] + \
                                    nDir[1] < Agent.height:
                                Agent.grid[pos[0] + dir[0] + nDir[0]][
                                    pos[1] + dir[1] + nDir[1]].enemy = True
                elif posLen < 10:
                    for dir in directions:
                        if -1 < pos[0] + dir[0] < Agent.width and -1 < pos[1] + dir[1] < Agent.height:
                            Agent.grid[pos[0] + dir[0]][pos[1] + dir[1]].enemy = True


    # Calculate manhattan distance
    def getManhattanDistance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0])+abs(pos1[1] - pos2[1])

    def initFood(self, gameState):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        redFood = gameState.getRedFood()
        for x in range(int(self.width / 2)):
            for y in range(self.height):
                if not redFood[x][y]:
                    Agent.grid[x][y].food = 0
                    continue
                food = 1
                # Check straight steps
                for dir in directions:
                    step = 0
                    while step < 2:
                        step += 1
                        if Agent.grid[x + dir[0] * step][y + dir[1] * step].wall:
                            break
                        if redFood[x + dir[0] * step][y + dir[1] * step]:
                            food += 1
                # Check diagonals
                for i in range(4):
                    iNext = i + 1 if i < 3 else 0
                    if redFood[x + directions[i][0] + directions[iNext][0]][y + directions[i][1] + directions[iNext][1]]:
                        if not Agent.grid[x + directions[i][0]][y + directions[i][1]].wall or not Agent.grid[x + directions[iNext][0]][y + directions[iNext][1]].wall:
                            food += 1
                Agent.grid[x][y].food = food
        # Mirror foods to blue side
        blueFood = gameState.getBlueFood()
        for x in range(int(self.width / 2), self.width):
            for y in range(self.height):
                if not blueFood[x][y]:
                    Agent.grid[x][y].food = 0
                    continue
                food = 1
                # Check straight steps
                for dir in directions:
                    step = 0
                    while step < 2:
                        step += 1
                        if Agent.grid[x + dir[0] * step][y + dir[1] * step].wall:
                            break
                        if blueFood[x + dir[0] * step][y + dir[1] * step]:
                            food += 1
                # Check diagonals
                for i in range(4):
                    iNext = i + 1 if i < 3 else 0
                    if blueFood[x + directions[i][0] + directions[iNext][0]][
                        y + directions[i][1] + directions[iNext][1]]:
                        if not Agent.grid[x + directions[i][0]][y + directions[i][1]].wall or not \
                        Agent.grid[x + directions[iNext][0]][y + directions[iNext][1]].wall:
                            food += 1
                Agent.grid[x][y].food = food


    def updateFood(self, pos):
        """
        Update food-cluster info in grid from food in pos being removed
        """
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        Agent.grid[pos[0]][pos[1]].food = 0
        # Update straight steps
        for dir in directions:
            step = 0
            while step < 2:
                step += 1
                if Agent.grid[pos[0] + dir[0] * step][pos[1] + dir[1] * step].wall:
                    break
                # Check that same side
                if (pos[0] < Agent.width and pos[0] + dir[0] * step >= Agent.width) or (pos[0] >= Agent.width and pos[0] + dir[0] * step < Agent.width):
                    break
                Agent.grid[pos[0] + dir[0] * step][pos[1] + dir[1] * step].food -= 1
        # Update diagonals
        for i in range(4):
            iNext = i + 1 if i < 3 else 0
            if not Agent.grid[pos[0] + directions[i][0] + directions[iNext][0]][pos[1] + directions[i][1] + directions[iNext][1]].wall:
                if not Agent.grid[pos[0] + directions[i][0]][pos[1] + directions[i][1]].wall or not \
                Agent.grid[pos[0] + directions[iNext][0]][pos[1] + directions[iNext][1]].wall:
                    # Check that same side
                    if (pos[0] < Agent.width and pos[0] + directions[i][0] + directions[iNext][0] >= Agent.width) or (
                            pos[0] >= Agent.width and pos[0] + directions[i][0] + directions[iNext][0] < Agent.width):
                        continue
                    Agent.grid[pos[0] + directions[i][0] + directions[iNext][0]][pos[1] + directions[i][1] + directions[iNext][1]].food -= 1


    def initCapsule(self, gameState):
        capsules = self.getCapsules(gameState)
        for pos in capsules:
            Agent.grid[pos[0]][pos[1]].capsule = True

    def countFood(self, gameState):
        """
        Count number of food left
        """
        foodGrid = self.getFood(gameState)
        count = 0
        for row in foodGrid:
            for node in row:
                if node:
                    count += 1
        return count


    # A* and related functions
    def findPath(self, startPos, goalPos, type='dist'):
        """
        Finds and returns the best path from startPos to goalPos along with it's value
        Value is based on type of path searched
        """
        start = Agent.grid[startPos[0]][startPos[1]]
        goal = Agent.grid[goalPos[0]][goalPos[1]]
        if start.wall or goal.wall:
            return [], 10000000
        openSet = [start]
        heapq.heapify(openSet)
        closedSet = set()
        while len(openSet) > 0:
            current = heapq.heappop(openSet)
            closedSet.add(current)
            if current is goal:
                break
            for neighbor in self.findNeighborNodes(current):
                if neighbor in closedSet:
                    continue
                self.updateNode(current, neighbor, goal, openSet, type)
        path = self.getPath(start, goal)
        cost = goal.gCost + goal.hCost
        self.resetCost()
        return path, cost

    def updateNode(self, current, neighbor, goal, openSet, type):
        """
        Updates the g- and h cost of neighbor with current as parent
        """
        getCost = self.getCostDist
        if type == 'food':
            getCost = self.getCostFood
        elif type == 'flee':
            getCost = self.getCostFlee
        elif type == 'capsule':
            getCost = self.getCostCapsule
        elif type == 'chase':
            getCost = self.getCostChase
        elif type == 'defend':
            getCost = self.getCostDefend
        elif type == 'noisydistance':
                getCost = self.getCostNoisyChase
        elif type == 'support':
            getCost = self.getCostSupport
        elif type == 'hunt':
            getCost = self.getCostHunt    
        newG = current.gCost + getCost(current, neighbor)
        if newG < neighbor.gCost or neighbor not in openSet:
            neighbor.gCost = newG
            neighbor.hCost = self.getHCost(neighbor, goal)
            neighbor.prevPath = current
            if neighbor not in openSet:
                heapq.heappush(openSet, neighbor)

    def getCostDist(self, current, neighbor):
        """
        Returns cost of moving from current to neighbor
        """
        return 1

    def getCostFood(self, current, neighbor):
        """
        Cost of moving from current to neighbor when eating food
        """
        cost = max(1, 10 - neighbor.food)
        if neighbor.enemy:
            cost += 50
        elif neighbor.capsule:
            cost += 50
        if neighbor.friend:
            cost += 30
        return cost

    def getCostFlee(self, current, neighbor):
        """
        Cost of moving from current to neighbor when eating
        """
        cost = 2
        if neighbor.food > 0:
            cost -= 1
        if neighbor.enemy:
            cost += 16
        return cost

    def getCostChase(self, current, neighbor):
        """
        Cost of moving 1 step when chasing
        """
        cost = 10 + int(abs(neighbor.x - Agent.width/2)/2)
        if neighbor.enemy:
            cost -= 5
        if neighbor.choke:
            cost -= 3
        if neighbor.friend:
            cost += 20
        return max(1, cost)


    def getCostCapsule(self, current, neighbor):
        """
        Cost of moving from current to neighbor when powerUp
        """
        cost = max(1, 20 - neighbor.food * 2)
        cost += int(Agent.width / 2 - abs(Agent.width / 2 - neighbor.x))
        if neighbor.food > 0:
            cost += Agent.width - abs(int(Agent.width / 2) - neighbor.x)
        if neighbor.friend:
            cost += 30
        if neighbor.capsule:
            cost += 100
        return cost

    def getCostDefend(self, current, neighbor):
        """
        Cost of moving 1 step when defending
        """
        return 15
    
    def getCostNoisyChase(self, current, neighbor):
        """
        Cost of moving from border during noisy chase
        """
        pos_x = neighbor.x
        cost = abs(pos_x - self.width/2)
        return cost
    
    def getCostHunt(self, current, neighbor):
        """
        Cost of hunting
        """
        cost = 10
        if neighbor.enemy and not self.homeSide:
            cost += 30
        elif neighbor.enemy and self.scared:
            cost += 25
        elif neighbor.enemy and self.homeSide:
            cost -= 10
        return max(1, cost)
    
    def getCostSupport(self, current, neighbor):
        """
        Cost of moving for supporting
        """
        cost = 10 + int(abs(neighbor.x - Agent.width/2)/2)
        if neighbor.enemy:
            cost -= 5
        if neighbor.choke == Agent.supportChoke:
            cost -= 5
        if neighbor.friend:
            cost += 10
        return max(1, cost)

    def getHCost(self, neighbor, goal):
        """
        Returns approximated cost from neighbor to goal
        Currently returns manhattan distance
        """
        return self.getManhattanDistance((neighbor.x, neighbor.y), (goal.x, goal.y))

    def resetCost(self):
        """
        Reset all costs of nodes
        """
        for x in range(Agent.width):
            for y in range(Agent.height):
                Agent.grid[x][y].hCost = 0
                Agent.grid[x][y].gCost = 0


    def getPath(self, start, goal):
        """
        Returns list of nodes which constitutes the ebst path from start to goal
        """
        path = []
        current = goal
        while current is not start:
            path.append(current)
            current = current.prevPath
        path.reverse()
        return path

    def findNeighborNodes(self, node):
        """
        Finds neighboring non wall nodes to node
        """
        neighborPos = self.findNeighbors(node.x, node.y)
        neighborNodes = set()
        for pos in neighborPos:
            neighborNodes.add(Agent.grid[pos[0]][pos[1]])
        return neighborNodes

    def updateOwnAction(self, gameState, action):
        """
        Update grid and power up based on own action
        """
        self.turnTimer -= 1
        currentPos = gameState.getAgentPosition(self.index)
        move = self.getMove(action)
        nextPos = (currentPos[0] + move[0], currentPos[1] + move[1])
        if (self.red and nextPos[0] >= Agent.width / 2) or (not self.red and nextPos[1] < Agent.width / 2):
            self.homeSide = False
            if Agent.grid[nextPos[0]][nextPos[1]].food > 0:
                self.updateFood(nextPos)
            if Agent.grid[nextPos[0]][nextPos[1]].capsule:
                Agent.grid[nextPos[0]][nextPos[1]].capsule = False
        else:
            self.homeSide = True
        # Update enemy pos if killed
        if Agent.grid[nextPos[0]][nextPos[1]].enemy:
            for i in range(2):
                if len(Agent.enemyPos[i]) == 1:
                    for pos in Agent.enemyPos[i]:
                        if pos == nextPos:
                            Agent.death = True
                            if self.homeSide and not self.scared:
                                Agent.enemyPos[i].clear()
                                Agent.enemyPos[i].add(gameState.getInitialAgentPosition(Agent.enemyIndex[i]))
                            elif not self.homeSide and self.powerUp:
                                Agent.enemyPos[i].clear()
                                Agent.enemyPos[i].add(gameState.getInitialAgentPosition(Agent.enemyIndex[i]))
        # Update friend sphere of influence
        for x in range(Agent.width):
            for y in range(Agent.height):
                Agent.grid[x][y].friend = False
        Agent.grid[nextPos[0]][nextPos[1]].friend = True
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i, dir in enumerate(directions):
            if Agent.grid[nextPos[0] + dir[0]][nextPos[1] + dir[1]].wall:
                continue
            nextDirs = [(0, 0), directions[i - 1 if i > -1 else 3], dir, directions[i + 1 if i < 3 else 0]]
            for nDir in nextDirs:
                if -1 < nextPos[0] + dir[0] + nDir[0] < Agent.width and -1 < nextPos[1] + dir[1] + nDir[1] < Agent.height:
                    Agent.grid[nextPos[0] + dir[0] + nDir[0]][nextPos[1] + dir[1] + nDir[1]].friend = True


    def updatePowerUp(self, gameState):
        """
        Checks if powerupped
        """
        scaredTimers = [gameState.getAgentState(Agent.enemyIndex[0]).scaredTimer, gameState.getAgentState(Agent.enemyIndex[1]).scaredTimer]
        if scaredTimers[0] == 0 and scaredTimers[1] == 0:
            self.powerUp = False
        elif scaredTimers[0] > 0 and scaredTimers[1] > 0:
            self.powerUp = True
            self.powerTimer = scaredTimers[0]
        else:
            self.powerUp = True
            if self.powerTimer > 10:
                self.powerTimer = 10
            self.powerTimer -= 1
            if self.powerTimer < 1:
                self.powerUp = False

    def updateScared(self, gameState):
        """
        Check if scared
        """
        scaredTimer = gameState.getAgentState(self.index).scaredTimer
        if scaredTimer > 0:
            self.scared = True
        else:
            self.scared = False

    def actionEat(self, gameState):
        """
        Find best action when eating
        Transitions to flee
        """
        currentFood = gameState.getAgentState(self.index).numCarrying
        currentPos = gameState.getAgentPosition(self.index)
        food = self.countFood(gameState)
        enemyFood = sum(Agent.enemyFood)
        approxExitDist = abs(currentPos[0] - Agent.width / 2) * 2
        # If winning amount of food carried escape
        if food < 3:
            Agent.friendState[self.teamIndex] = self.States.Flee
            return self.actionFlee(gameState)
        if approxExitDist > self.turnTimer:
            Agent.friendState[self.teamIndex] = self.States.Flee
            return self.actionFlee(gameState)
        if food < enemyFood and Agent.friendState[1 - self.teamIndex] == self.States.Eat:
            Agent.friendState[self.teamIndex] = self.States.Chase
            return self.actionChase(gameState)
        if self.powerUp:
            Agent.friendState[self.teamIndex] = self.States.Capsule
            return self.actionCapsule(gameState)
        if (self.red and currentPos[0] < Agent.width / 2) or (
                not self.red and currentPos[0] >= Agent.width / 2):
            Agent.friendState[self.teamIndex] = self.States.Choose
            return self.actionChoose(gameState)
        # Compare food carried with how close enemy is, flee if risky
        # Get approximation of enemy vicinity and exits blocked
        enemyDist = []
        for i in range(2):
            closestManhattanDist = 100
            closestManhattanPos = None
            averageManhattanDist = 0
            for pos in Agent.enemyPos[i]:
                dist = self.getManhattanDistance(pos, currentPos)
                if dist < closestManhattanDist:
                    closestManhattanPos = pos
                    closestManhattanDist = dist
                averageManhattanDist += dist
            averageManhattanDist = averageManhattanDist / len(Agent.enemyPos[i])
            _, closestManhattanDist = self.findPath(closestManhattanPos, currentPos)
            enemyDist.append(min(closestManhattanDist, averageManhattanDist))

        exitDist = []
        for exit in Agent.grid[currentPos[0]][currentPos[1]].exits:
            _, dist = self.findPath(currentPos, exit)
            exitDist.append(dist)

        exitCount = len(exitDist)
        fleeDist = 3
        if currentFood > 0:
            fleeDist = 5
        if min(enemyDist) < fleeDist:
            Agent.friendState[self.teamIndex] = self.States.Flee
            return self.actionFlee(gameState)
        if exitCount == 1:
            if min(enemyDist) < exitDist[0] * 2 + currentFood / 2:
                Agent.friendState[self.teamIndex] = self.States.Flee
                return self.actionFlee(gameState)
        elif exitCount == 2:
            if enemyDist[0] < min(exitDist) * 2 + currentFood / 2 and enemyDist[1] < min(exitDist) * 2 + currentFood / 2:
                Agent.friendState[self.teamIndex] = self.States.Flee
                return self.actionFlee(gameState)
        # Compare value of eating food
        # First vacuum up all food within k = 2 steps
        bestDir = None
        bestFood = 0
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i, dir in enumerate(directions):
            if Agent.grid[currentPos[0] + dir[0]][currentPos[1] + dir[1]].wall or Agent.grid[currentPos[0] + dir[0]][currentPos[1] + dir[1]].capsule:
                continue
            nextDirs = [(0, 0), directions[i - 1 if i > -1 else 3], dir, directions[i + 1 if i < 3 else 0]]
            for nDir in nextDirs:
                if Agent.grid[currentPos[0] + dir[0] + nDir[0]][currentPos[1] + dir[1] + nDir[1]].food > bestFood:
                    bestFood = Agent.grid[currentPos[0] + dir[0] + nDir[0]][currentPos[1] + dir[1] + nDir[1]].food
                    bestDir = i
        if bestFood > 0:
            return Agent.actions[bestDir]
        # Compare food clusters
        bestAction = ''
        bestCost = 10000
        if self.red:
            xRange = int(Agent.width / 2), Agent.width, 1
        else:
            xRange = int(Agent.width / 2) - 1, -1, -1
        for x in range(xRange[0], xRange[1], xRange[2]):
            for y in range(self.height):
                if Agent.grid[x][y].food > 0:
                    path, cost = self.findPath(currentPos, (x, y), 'food')
                    if cost < bestCost:
                        bestCost = cost
                        bestAction = self.posToAction(currentPos, (path[0].x, path[0].y))
        return bestAction


    def actionFlee(self, gameState):
        """
        Find best action when fleeing
        Transition into capsule or choose
        """
        currentPos = gameState.getAgentPosition(self.index)
        # If on own side change to choose, if eaten capsule change to capsule
        if (self.red and currentPos[0] < Agent.width / 2) or (not self.red and currentPos[0] >= Agent.width / 2):
            Agent.friendState[self.teamIndex] = self.States.Choose
            return self.actionChoose(gameState)
        if self.powerUp and self.countFood(gameState) > 2:
            Agent.friendState[self.teamIndex] = self.States.Capsule
            return self.actionCapsule(gameState)
        # Compare all exits
        bestCost = 10000
        bestPath = None
        # Compare exits
        exits = Agent.grid[currentPos[0]][currentPos[1]].exits
        for exit in exits:
            exitPath, exitCost = self.findPath(currentPos, exit, 'flee')
            escapePath, escapeCost = self.findPath(exit, Agent.grid[exit[0]][exit[1]].scorePos, 'flee')
            if exitCost + escapeCost < bestCost:
                if len(exitPath) > 0:
                    bestCost = exitCost + escapeCost
                    bestPath = exitPath
                elif len(escapePath) > 0:
                    bestCost = exitCost + escapeCost
                    bestPath = escapePath
        # Compare capsules
        for pos in self.getCapsules(gameState):
            capPath, capCost = self.findPath(currentPos, pos, 'flee')
            if capCost < bestCost and len(capPath) > 0:
                bestCost = capCost
                bestPath = capPath

        return self.posToAction(currentPos, (bestPath[0].x, bestPath[0].y))

    def actionCapsule(self, gameState):
        """
        If eating and power up
        """
        currentPos = gameState.getAgentPosition(self.index)
        if (self.red and currentPos[0] < Agent.width / 2) or (not self.red and currentPos[0] >= Agent.width / 2):
            Agent.friendState[self.teamIndex] = self.States.Choose
            return self.actionChoose(gameState)
        if self.countFood(gameState) < 3:
            Agent.friendState[self.teamIndex] = self.States.Flee
            return self.actionFlee(gameState)
        if not self.powerUp:
            Agent.friendState[self.teamIndex] = self.States.Eat
            return self.actionEat(gameState)
        if self.red:
            possibleExit = (int(Agent.width / 2) - 1, currentPos[1])
        else:
            possibleExit = (int(Agent.width / 2), currentPos[1])
        step = 0
        while True:
            if possibleExit[1] + step < Agent.height:
                if not Agent.grid[possibleExit[0]][possibleExit[1] + step].wall:
                    possibleExit = (possibleExit[0], possibleExit[1] + step)
                    break
            if possibleExit[1] - step > -1:
                if not Agent.grid[possibleExit[0]][possibleExit[1] - step].wall:
                    possibleExit = (possibleExit[0], possibleExit[1] - step)
                    break
            step += 1
        exitPath, exitDist = self.findPath(currentPos, possibleExit, 'dist')
        # Get estimated enemy positions
        estimatedEnemyPos = []
        estimatedEnemyDist = []
        for i in range(2):
            estimatedX = 0
            estimatedY = 0
            for pos in Agent.enemyPos[i]:
                estimatedX += pos[0]
                estimatedY += pos[1]
            estimatedPos = (int(estimatedX / len(Agent.enemyPos[i])), int(estimatedY / len(Agent.enemyPos[i])))
            while True:
                if Agent.grid[estimatedPos[0]][estimatedPos[1]].wall:
                    estimatedPos = (estimatedPos[0] + np.sign(int(Agent.width / 2) - estimatedPos[0]), estimatedPos[1] + np.sign(int(Agent.height / 2) - estimatedPos[1]))
                else:
                    break
            estimatedEnemyPos.append(estimatedPos)
            _, dist = self.findPath(currentPos, estimatedPos)
            estimatedEnemyDist.append(dist)
        if exitDist > self.powerTimer - 2 and exitDist > max(estimatedEnemyDist):
            return self.posToAction(currentPos, (exitPath[0].x, exitPath[0].y))
        if exitDist + 3 > self.turnTimer:
            return self.posToAction(currentPos, (exitPath[0].x, exitPath[0].y))
        # First vacuum up all food within k = 2 steps
        bestDir = None
        bestFood = 0
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i, dir in enumerate(directions):
            if Agent.grid[currentPos[0] + dir[0]][currentPos[1] + dir[1]].wall or Agent.grid[currentPos[0] + dir[0]][currentPos[1] + dir[1]].capsule:
                continue
            nextDirs = [(0, 0), directions[i - 1 if i > -1 else 3], dir, directions[i + 1 if i < 3 else 0]]
            for nDir in nextDirs:
                if Agent.grid[currentPos[0] + dir[0] + nDir[0]][currentPos[1] + dir[1] + nDir[1]].food > bestFood:
                    bestFood = Agent.grid[currentPos[0] + dir[0] + nDir[0]][currentPos[1] + dir[1] + nDir[1]].food
                    bestDir = i
        if bestFood > 0:
            return Agent.actions[bestDir]
        # Else check if available food elsewhere
        # Compare food clusters
        bestAction = ''
        bestCost = 100000
        if self.red:
            xRange = int(Agent.width / 2), Agent.width, 1
        else:
            xRange = int(Agent.width / 2) - 1, -1, -1
        for x in range(xRange[0], xRange[1], xRange[2]):
            for y in range(self.height):
                if Agent.grid[x][y].food > 0:
                    path, cost = self.findPath(currentPos, (x, y), 'capsule')
                    if cost < bestCost:
                        bestCost = cost
                        bestAction = self.posToAction(currentPos, (path[0].x, path[0].y))
        return bestAction


    def actionChoose(self, gameState):
        """
        Choose whether to chase or eat
        """
        currentPos = gameState.getAgentPosition(self.index)
        # Compare food clusters
        bestActionEat = ''
        bestCostEat = 100000
        if self.red:
            xRange = int(Agent.width / 2), Agent.width, 1
        else:
            xRange = int(Agent.width / 2) - 1, -1, -1
        for x in range(xRange[0], xRange[1], xRange[2]):
            for y in range(self.height):
                if Agent.grid[x][y].food > 0:
                    approxCost = self.getManhattanDistance(currentPos, (x, y)) * 20
                    if approxCost > bestCostEat:
                        continue
                    path, cost = self.findPath(currentPos, (x, y), 'food')
                    if cost < bestCostEat:
                        bestCostEat = cost
                        bestActionEat = self.posToAction(currentPos, (path[0].x, path[0].y))
        # Get estimated enemy positions
        estimatedEnemyPos = []
        estimatedEnemyDist = []
        for i in range(2):
            estimatedX = 0
            estimatedY = 0
            for pos in Agent.enemyPos[i]:
                estimatedX += pos[0]
                estimatedY += pos[1]
            estimatedPos = (int(estimatedX / len(Agent.enemyPos[i])), int(estimatedY / len(Agent.enemyPos[i])))
            while True:
                if Agent.grid[estimatedPos[0]][estimatedPos[1]].wall:
                    estimatedPos = (estimatedPos[0] + np.sign(int(Agent.width / 2) - estimatedPos[0]), estimatedPos[1] + np.sign(int(Agent.height / 2) - estimatedPos[1]))
                else:
                    break
            estimatedEnemyPos.append(estimatedPos)
            _, dist = self.findPath(currentPos, estimatedPos)
            estimatedEnemyDist.append(dist)
        # Locked if enemy waiting on other side
        lockCount = 0
        if not self.scared and not self.powerUp:
            if not Agent.friendLocked[1 - self.teamIndex]:
                for i in range(2):
                    if not gameState.getAgentState(Agent.enemyIndex[i]).isPacman and not gameState.getAgentState(self.index).isPacman:
                        if abs(currentPos[0] - Agent.width/2) < 2 and estimatedEnemyDist[i] < 5:
                            lockCount += 1
                if lockCount > 0:
                    Agent.friendLocked[self.teamIndex] = True
                    noAction = 'East' if self.red else 'West'
                    actions = [act for act in gameState.getLegalActions(self.index) if act != noAction]
                    return random.choice(actions)
                else:
                    Agent.friendLocked[self.teamIndex] = False
            # Move away from lock if friend is already locked
            else:
                lockedEnemyPos = (1, 1)
                for i in range(2):
                    if not gameState.getAgentState(Agent.enemyIndex[i]).isPacman and not gameState.getAgentState(self.index).isPacman:
                        if abs(currentPos[0] - Agent.width/2) < 2 and estimatedEnemyDist[i] < 5:
                            lockCount += 1
                            lockedEnemyPos = estimatedEnemyPos[i]
                if lockCount > 0:
                    yRange = 0, Agent.height, 1
                    if lockedEnemyPos[1] < Agent.height / 2:
                        yRange = Agent.height - 1, -1, -1
                    for y in range(yRange[0], yRange[1], yRange[2]):
                        if not Agent.grid[currentPos[0]][y].wall:
                            path, _ = self.findPath(currentPos, (currentPos[0], y))
                            if len(path) > 0:
                                if len(path) < 3:
                                    break
                                else:
                                    return self.posToAction(currentPos, (path[0].x, path[0].y))
        else:
            Agent.friendLocked[self.teamIndex] = False
        # Compare chase actions
        bestActionChase = ''
        bestCostChase = 100000
        for i in range(2):
            if not gameState.getAgentState(Agent.enemyIndex[i]).isPacman:
                continue
            path, cost = self.findPath(currentPos, estimatedEnemyPos[i], 'chase')
            if cost - Agent.enemyFood[i] * 20 < bestCostChase:
                bestCostChase = cost - Agent.enemyFood[i] * 20
                bestActionChase = self.posToAction(currentPos, (path[0].x, path[0].y))
        # Add compare defend actions
        bestActionDefend = ''
        bestCostDefend = 100000
        if self.red:
            xRange = 0, int(Agent.width/2)
        else:
            xRange = int(Agent.width/2), Agent.width
        for x in range(xRange[0], xRange[1]):
            for y in range(Agent.height):
                food = Agent.grid[x][y].food
                if food > 2:
                    enemyCost = 10000
                    for i in range(2):
                        approxCost = self.getManhattanDistance(estimatedEnemyPos[i], (x, y)) * 20
                        if approxCost > bestCostDefend:
                            continue
                        _, cost = self.findPath(estimatedEnemyPos[i], (x, y), 'defend')
                        if cost < enemyCost:
                            enemyCost = cost
                    path, selfCost = self.findPath(currentPos, (x, y), 'defend')
                    if selfCost < enemyCost + food * 10 and enemyCost < bestCostDefend and len(path) > 0:
                        bestCostDefend = enemyCost
                        bestActionDefend = self.posToAction(currentPos, (path[0].x, path[0].y))

        # print(bestCostEat, bestCostChase, bestCostDefend)
        # If scared move towards border with intent to eat
        if self.scared:
            if not self.homeSide:
                Agent.friendState[self.teamIndex] = self.States.Eat
            return bestActionEat
        if bestCostEat < bestCostChase and bestCostEat < bestCostDefend and self.countFood(gameState) > 2:
            if not self.homeSide:
                Agent.friendState[self.teamIndex] = self.States.Eat
            return bestActionEat
        elif bestCostChase < bestCostEat and bestCostChase < bestCostDefend:
            if self.homeSide:
                Agent.friendState[self.teamIndex] = self.States.Chase
            return bestActionChase
        else:
            return bestActionDefend


    def actionChase(self, gameState):
        """
        Chase down pacman if close enough and in our own map
        """
        selfpos = gameState.getAgentPosition(self.index)
        teampos = gameState.getAgentPosition(1-self.teamIndex)
        val = [0,0]

        
        #if enemy1 food > 2, look enemy1, vice versa
        checkpos = []
        count = 0
        for enemyindex in range(len(self.enemyIndex)):
            if not gameState.getAgentState(Agent.enemyIndex[enemyindex]).isPacman:
                count += 1
            if len(Agent.enemyPos[enemyindex]) > 1:
                estimatedX = 0
                estimatedY = 0
                for pos in Agent.enemyPos[enemyindex]:
                    estimatedX += pos[0]
                    estimatedY += pos[1]
                estimatedPos = (int(estimatedX / len(Agent.enemyPos[enemyindex])), int(estimatedY / len(Agent.enemyPos[enemyindex])))
                while True:
                    if Agent.grid[estimatedPos[0]][estimatedPos[1]].wall:
                        estimatedPos = (estimatedPos[0] + np.sign(int(Agent.width / 2) - estimatedPos[0]), estimatedPos[1] + np.sign(int(Agent.height / 2) - estimatedPos[1]))
                    else:
                        break
                checkpos.append(estimatedPos)
            else:
                for pos in Agent.enemyPos[enemyindex]:
                  checkpos.append(pos)
            #print(enemyindex)
            val[enemyindex]  = abs(Agent.enemyFood[enemyindex]*2 + round(10/self.getManhattanDistance(selfpos, checkpos[enemyindex])))
            
        if count == 2:
            Agent.friendState[self.teamIndex] = self.States.Choose
            count = 0
            return self.actionChoose(gameState)
        
        max_val = max(val)
        enemy_index = val.index(max_val)
        Agent.targetIndex = enemy_index             
        enemyposset = Agent.enemyPos[enemy_index]
        if len(enemyposset) > 1:
            noisydistance = True
        else:
            for enemypos in enemyposset:
              break
            noisydistance = False
        

        #find good path to noisy distance, while not trying to move too far away from border
        if noisydistance:
            # Get estimated enemy positions
            path, value = self.findPath(selfpos, checkpos[enemyindex], 'noisydistance')
            if path != None and len(path)>0:
                return self.posToAction(selfpos, (path[0].x, path[0].y)) 
            #go near mass center, but keep close to border
            
   
        #if known distance then try to choke opponent in place and then kill    
        else:
            _, value1 = self.findPath(selfpos, (int(Agent.width/2), enemypos[1]), 'chase')
            _, value2 = self.findPath(enemypos, (int(Agent.width/2), enemypos[1]), 'chase')
            
            if value2> value1 and not self.scared:
                #check if segment has more than one chokepoint
                if len(Agent.grid[enemypos[0]][enemypos[1]].exits) > 1:
                    bestpathOne, previousvalue = self.findPath(selfpos, enemypos, 'hunt')
                    bestchokeOne = None
                    
                    #check closest point
                    for chokepoint in Agent.grid[enemypos[0]][enemypos[1]].exits:
                        path, value = self.findPath(selfpos, chokepoint, 'hunt')
                        if bestchokeOne is None:
                          bestchokeOne = chokepoint 
                        if value1 < previousvalue and chokepoint and Agent.grid[enemypos[0]][enemypos[1]].exits != Agent.grid[selfpos[0]][selfpos[1]].exits:
                            #the choke point that is closest
                            bestchokeOne = chokepoint
                            bestpathOne = path
                            bestpathOne.extend(self.findPath(bestchokeOne, enemypos, 'hunt'))
                        previousvalue = value1
                    #how to send pac there? path list to movement function?
                    _, value1 = self.findPath(enemypos, bestchokeOne)
                    _, value2 = self.findPath(selfpos, bestchokeOne)
                    if value1 >= value2 and teampos:
                        bestpathTwo, previousvalue = self.findPath(teampos, enemypos, 'hunt')
                        bestchokeTwo = None
                        for chokepoint in Agent.grid[enemypos[0]][enemypos[1]].exits:
                            path, value = self.findPath(teampos, chokepoint, 'hunt')
                            if bestchokeTwo is None:
                              bestchokeTwo = chokepoint
                            if value < previousvalue and chokepoint != bestchokeOne and Agent.grid[enemypos[0]][enemypos[1]].exits != Agent.grid[selfpos[0]][selfpos[1]].exits:
                                #the choke point that is closest and isnt the same chokepoint as the first one
                                #tell second to go to support function
                                bestchokeTwo = chokepoint
                                bestpathTwo = path
                                bestpathTwo.extend(self.findPath(bestchokeTwo, enemypos, 'hunt'))
                            previousvalue = value1
                            Agent.supportChoke = bestchokeTwo
                        _, value1 = self.findPath(enemypos, (int(Agent.width/2), enemypos[1]))
                        _, value2 = self.findPath(teampos, (int(Agent.width/2), enemypos[1]))
                        if value1 >= value2 and Agent.friendState[(1-self.teamIndex)] is not (self.States.Flee or self.States.Capsule) and Agent.enemyFood[Agent.targetIndex] > 6:
                            
                            #Agent.supportPath = bestpathTwo
                            Agent.friendState[(1-self.teamIndex)]  = Agent.States.Support
                            #self.actionSupport(gameState)
                    if bestpathOne != None and len(bestpathOne)>0:
                        return self.posToAction(selfpos, (bestpathOne[0].x, bestpathOne[0].y)) 
                else:
                    for c in Agent.grid[enemypos[0]][enemypos[1]].exits:
                      break
                    path, _ = self.findPath(selfpos, enemypos)
                    if Agent.grid[enemypos[0]][enemypos[1]].exits != Agent.grid[selfpos[0]][selfpos[1]].exits:
                        path1, _ = self.findPath(selfpos, c)
                        path2, _ = self.findPath(c, enemypos)
                        path = path1 + path2
                    if path != None and len(path)>0:
                        return self.posToAction(selfpos, (path[0].x, path[0].y)) 
            else:
                Agent.friendState[self.teamIndex] = self.States.Choose
                return self.actionChoose(gameState)
        



#kolla dist i alla exits, kolla avstånden till gränsen

    def actionSupport(self, gameState):
        """
        Support friendly pac
        """
        if self.scared or Agent.enemyFood[Agent.targetIndex] < 7:
            Agent.friendState[self.teamIndex] = self.States.Choose
            return self.actionChoose(gameState)
        currentPos = gameState.getAgentPosition(self.index)
        # Get estimated enemy positions
        estimatedEnemyPos = [(-1, -1), (-1, -1)]
        count = 0
        for i in range(2):
            if not gameState.getAgentState(Agent.enemyIndex[i]).isPacman:
                count += 1
                continue
            estimatedPos = (0, 0)
            for pos in Agent.enemyPos[i]:
                estimatedPos = (estimatedPos[0] + pos[0], estimatedPos[1] + pos[1])
            estimatedPos = (int(estimatedPos[0] / len(Agent.enemyPos[i])), int(estimatedPos[1] / len(Agent.enemyPos[i])))
            while True:
                if Agent.grid[estimatedPos[0]][estimatedPos[1]].wall:
                    estimatedPos = (estimatedPos[0] + np.sign(int(Agent.width / 2) - estimatedPos[0]), estimatedPos[1] + np.sign(int(Agent.height / 2) - estimatedPos[1]))
                else:
                    break
            estimatedEnemyPos[i] = estimatedPos
        if count == 2:
            Agent.friendState[self.teamIndex] = self.States.Choose
            return self.actionChoose(gameState)
        bestPath = None
        bestCost = 100000
        
        
        for i, pos in enumerate(estimatedEnemyPos):
            if pos == (-1, -1):
                continue
            path, cost = self.findPath(currentPos, pos, 'support')
            if cost - Agent.enemyFood[i] * 10 < bestCost:
                bestCost = cost - Agent.enemyFood[i] * 10
                bestPath = path
        bestAction = self.posToAction(currentPos, (path[0].x, path[0].y))

        return bestAction

    def posToAction(self, pos1, pos2):
        """
        Return the action to move from pos1 to pos2
        """
        if self.getManhattanDistance(pos1, pos2) > 1:
            return None
        if pos1[0] < pos2[0]:
            return 'East'
        elif pos1[1] < pos2[1]:
            return 'North'
        elif pos1[0] > pos2[0]:
            return 'West'
        return 'South'

    def getMove(self, action):
        """
        Return tuple move related to action
        """
        if action == 'East':
            return 1, 0
        elif action == 'North':
            return 0, 1
        elif action == 'West':
            return -1, 0
        return 0, -1

    def chooseAction(self, gameState):
        """
        Function which returns your choice of move to the game
        """
        if Agent.death:
            Agent.death = False
            self.initFood(gameState)
        self.updateEnemyPositions(gameState)
        self.updatePowerUp(gameState)
        self.updateScared(gameState)
        action = None
        if Agent.friendState[self.teamIndex] == self.States.Choose:
            action = self.actionChoose(gameState)
        elif Agent.friendState[self.teamIndex] == self.States.Chase:
            action = self.actionChase(gameState)
        elif Agent.friendState[self.teamIndex] == self.States.Eat:
            action = self.actionEat(gameState)
        elif Agent.friendState[self.teamIndex] == self.States.Flee:
            action = self.actionFlee(gameState)
        elif Agent.friendState[self.teamIndex] == self.States.Capsule:
            action = self.actionCapsule(gameState)
        else:
            action = self.actionChoose(gameState)
            Agent.friendState[self.teamIndex] = self.States.Choose

        """
        self.debugClear()
        for pos in Agent.enemyPos[0]:
            self.debugDraw(pos, [228, 126, 31])
        for pos in Agent.enemyPos[1]:
            self.debugDraw(pos, [31, 228, 126])
        print(Agent.friendState[self.teamIndex])
        """
        legalActions = gameState.getLegalActions(self.index)
        if action not in legalActions:
            action = random.choice(legalActions)
        self.updateOwnAction(gameState, action)
        return action

    class States(enum.Enum):
        """
        States
        """
        Choose = 1
        Eat = 2
        Chase = 3
        Support = 4
        Flee = 5
        Capsule = 6

    class Node:
        x = 0
        y = 0
        gCost = 0
        hCost = 0
        prevPath = None
        wall = False
        choke = False
        scoreDist = 0
        scorePos = None
        exits = set()
        food = 0
        capsule = False
        enemy = False
        friend = False

        def __init__(self, ix, iy, isWall):
            self.x = ix
            self.y = iy
            self.wall = isWall

        def __eq__(self, other):
            return repr(self) == repr(other)

        def __lt__(self, other):
            if self.fCost() == other.fCost():
                return self.hCost < other.hCost
            return self.fCost() < other.fCost()

        def __hash__(self):
            return hash(repr(self))

        def fCost(self):
            return self.gCost + self.hCost
