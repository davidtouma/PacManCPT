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

from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import capture
from captureAgents import CaptureAgent
import util
import game
import numpy as np


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'DummyAgent', second = 'DummyAgent'):
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class BeliefState:

	beliefs = []

	@staticmethod
	def registerInitialState(gameState):
		for agent in range(gameState.getNumAgents()):
			BeliefState.beliefs.append(util.Counter())
			BeliefState.resetBeliefs(gameState, agent)

	@staticmethod
	def resetBeliefs(gameState, agent):
		width = gameState.getWalls().width
		height = gameState.getWalls().height

		initialPos = gameState.getInitialAgentPosition(agent)

		for x in range(width):
			for y in range(height):
				if (x, y) == initialPos:
					BeliefState.beliefs[agent][(x, y)] = 1.0
				else:
					BeliefState.beliefs[agent][(x, y)] = 0.0

	@staticmethod
	def updateBeliefs(gameState, captureAgent):

		prevState = captureAgent.getPreviousObservation()
		ownPos = gameState.getAgentPosition(captureAgent.index)
		noisyDistances = gameState.getAgentDistances()

		for agent in range(gameState.getNumAgents()):
			realPos = gameState.getAgentPosition(agent)
			if realPos:
				for pos in BeliefState.beliefs[agent]:
					BeliefState.beliefs[agent][pos] = 0.0
				BeliefState.beliefs[agent][realPos] = 1.0
				continue

			updatedBelief = util.Counter()

			for pos in BeliefState.beliefs[agent]:
				neighbours = getNeighbours(gameState, pos, True)
				neighbours = list(filter(lambda n: util.manhattanDistance(ownPos, n) > capture.SIGHT_RANGE, neighbours))

				for neighbour in neighbours:
					updatedBelief[neighbour] += BeliefState.beliefs[agent][pos] / len(neighbours)

			BeliefState.beliefs[agent] = updatedBelief

			probability = 0.0
			for pos in BeliefState.beliefs[agent]:
				trueDistance = util.manhattanDistance(ownPos, pos)
				probabilityGivenPos = gameState.getDistanceProb(trueDistance, noisyDistances[agent])
				probability += probabilityGivenPos * BeliefState.beliefs[agent][pos]

			if not probability:
				probability = 0.001

			for pos in BeliefState.beliefs[agent]:
				trueDistance = util.manhattanDistance(ownPos, pos)
				probabilityGivenPos = gameState.getDistanceProb(trueDistance, noisyDistances[agent])
				BeliefState.beliefs[agent][pos] = (probabilityGivenPos * BeliefState.beliefs[agent][pos]) / probability

			if prevState:
				prevRealPos = prevState.getAgentPosition(agent)
				if prevRealPos and util.manhattanDistance(ownPos, prevRealPos) <= 1:
					BeliefState.resetBeliefs(gameState, agent)
					continue

				food = list(captureAgent.getFoodYouAreDefending(gameState))
				prevFood = list(captureAgent.getFoodYouAreDefending(prevState))

				newlyEatenX, newlyEatenY = np.where(np.array(food) != np.array(prevFood))
				positions = list(zip(newlyEatenX, newlyEatenY))
				otherOpponent = list(filter(lambda index: agent != index, captureAgent.getOpponents(gameState)))[0]

				for foodPos in positions:
					trueDistance = util.manhattanDistance(ownPos, foodPos)
					probabilityGivenPos = gameState.getDistanceProb(trueDistance, noisyDistances[agent])
					otherProbabilityGivenPos = gameState.getDistanceProb(trueDistance, noisyDistances[otherOpponent])

					if probabilityGivenPos > 0 and otherProbabilityGivenPos == 0:
						for pos in BeliefState.beliefs[agent]:
							BeliefState.beliefs[agent][pos] = 0.0
						BeliefState.beliefs[agent][foodPos] = 1.0


class DummyAgent(CaptureAgent):

	targets = []
	opponentPosCache = None

	### OFFENSIVE ###

	def offensiveAction(self, gameState):
		""" Tree chunk for the offensive behaviour
			Can change to return at each step instead """

		ownPos = gameState.getAgentPosition(self.index)
		isScared = False
		safeSide = False
		isEscaping = False

		self.targets = [target for target in self.targets if target != ownPos]

		if self.isThreatened(gameState):
			capsulePos = self.getAvailableCapsule(gameState)

			self.targets = [capsulePos] if capsulePos or isEscaping else [self.escape(gameState)]

			if self.hasEscaped(gameState):  # Tries to position itself to not go straight into the opponents
				self.targets = [self.escape(gameState, margin=0)]  # Move towards

		elif self.isScared(gameState)[0]:
			isScared, index = self.isScared(gameState)
			self.targets = [self.huntScared(gameState, index)]

		elif not self.targets and self.onOpponentSide(gameState, ownPos):
			self.targets = [self.escape(gameState)]
		elif not self.targets:
			self.targets = self.vrp(gameState)

		path = self.aStar(gameState, ownPos, self.targets[0], not isScared)

		return self.pathToAction(gameState, path)

	def isThreatened(self, gameState):
		""" Opponents are within a certain range """

		ownPos = gameState.getAgentPosition(self.index)
		if not self.onOpponentSide(gameState, ownPos, margin=2):
			return False

		for opponent in self.getOpponents(gameState):
			opponentPos = gameState.getAgentPosition(opponent)
			opponentIsScared = gameState.getAgentState(opponent).scaredTimer > 10

			if opponentPos and self.getMazeDistance(ownPos, opponentPos) < 5 and not opponentIsScared:
				if self.onOpponentSide(gameState, opponentPos): # Added
					self.opponentPosCache = opponentPos
					return True

		return False

	def isScared(self, gameState):
		""" If opponent is scared and close, hunt opponent instead of food """


		for opponent in self.getOpponents(gameState):
			opponentIsScared = gameState.getAgentState(opponent).scaredTimer > 5	# Margin

			if opponentIsScared:	# Saves computation having it here
				ownPos = gameState.getAgentPosition(self.index)
				opponentPos = gameState.getAgentPosition(opponent)

				if opponentPos and self.getMazeDistance(ownPos, opponentPos) < 5:	# If opponent is visible within 5
					return True, opponent	# Will randomly pick some close ghost, returns pos + index
		extra = 0
		return False, extra

	def huntScared(self, gameState, opponentIndex):
		""" Mostly for clearer behaviour. Returns position of close scared ghost """
		opponentPos = gameState.getAgentPosition(opponentIndex)
		return opponentPos

	def escape(self, gameState, margin=2):	# Use margin 0 to move along safe side when threatened
		""" Finds random point behind halfpoint to escape to """

		width = gameState.getWalls().width
		height = gameState.getWalls().height

		halfwayPoint = width // 2 - margin if self.red else width // 2 + margin	# Added safety margin

		opponentPos = self.opponentPosCache
		heightHalfPoint = height / 2

		while True:
			if opponentPos:
				escapePoint = opponentPos[1]	# Projection to halfline
			else:
				escapePoint = height - 1

			if escapePoint > heightHalfPoint:
				row = np.random.randint(1, heightHalfPoint)
			else:
				row = np.random.randint(heightHalfPoint, height-1)

			if not gameState.hasWall(halfwayPoint, row):

				return (halfwayPoint, row)

	def hasEscaped(self, gameState):
		""" Check if escaped. This check is only run after an escape is completed """
		ownPos = gameState.getAgentPosition(self.index)
		return ownPos == self.targets[0]

	def getAvailableCapsule(self, gameState):
		""" If safe path to capsule. """

		for capsulePos in self.getCapsules(gameState):
			for opponent in self.getOpponents(gameState):
				if BeliefState.beliefs[opponent][capsulePos] > 0.05:  # If possible pacman there
					break
			else:
				return capsulePos

		return None

	def vrp(self, gameState):
		nodes, distanceMatrix = self.generateDistanceMatrix(gameState)
		demands = [0] + [1] * (len(distanceMatrix) - 1)

		data = {
			'distance_matrix': distanceMatrix,
			'demands': demands,
			'vehicle_capacities': [7],
			'num_vehicles': 1,
			'depot': 0
		}

		manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
		routing = pywrapcp.RoutingModel(manager)

		def distanceCallback(from_index, to_index):
			from_node = manager.IndexToNode(from_index)
			to_node = manager.IndexToNode(to_index)
			return data['distance_matrix'][from_node][to_node]

		transitCallbackIndex = routing.RegisterTransitCallback(distanceCallback)
		routing.SetArcCostEvaluatorOfAllVehicles(transitCallbackIndex)

		def demandCallback(from_index):
			from_node = manager.IndexToNode(from_index)
			return data['demands'][from_node]

		demandCallbackIndex = routing.RegisterUnaryTransitCallback(demandCallback)
		routing.AddDimensionWithVehicleCapacity(demandCallbackIndex, 0, data['vehicle_capacities'], True, 'Capacity')

		for node in range(1, len(data['distance_matrix'])):
			penalty = data['vehicle_capacities'][0] + getPenalty(gameState, nodes[node]) + 10
			routing.AddDisjunction([manager.NodeToIndex(node)], int(penalty))

		searchParameters = pywrapcp.DefaultRoutingSearchParameters()
		searchParameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
		searchParameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
		searchParameters.time_limit.seconds = 1
		searchParameters.lns_time_limit.seconds = 1
		searchParameters.solution_limit = 100
		searchParameters.log_search = True

		solution = routing.SolveWithParameters(searchParameters)

		if solution:
			index = routing.Start(0)
			route = []
			while not routing.IsEnd(index):
				nodeIndex = manager.IndexToNode(index) - 1
				if nodeIndex > 0:
					route.append(nodes[nodeIndex])
				index = solution.Value(routing.NextVar(index))
			return route
		else:
			return []

	def generateDistanceMatrix(self, gameState):
		distances = []

		foodX, foodY = np.nonzero(np.array(list(self.getFood(gameState))))
		nodes = list(zip(foodX, foodY))
		nodes.insert(0, gameState.getAgentPosition(self.index))

		for i, node1 in enumerate(nodes):
			distances.append([])
			for node2 in nodes:
				distances[i].append(self.getMazeDistance(node1, node2))

		return nodes, distances

	### OFFENSIVE END ###

	### DEFENSIVE ###

	def defensiveAction(self, gameState):

		ownPos = gameState.getAgentPosition(self.index)

		self.targets = [target for target in self.targets if target != ownPos]

		if gameState.getAgentState(self.index).isPacman:
			self.targets = [self.escape(gameState)]
		else:
			invaders = self.getInvaders(gameState)

			if invaders:
				self.targets = [self.chase(gameState, invaders)]
			elif not self.targets:
				self.targets = self.getPatrolPositions(gameState)

		path = self.aStar(gameState, ownPos, self.targets[0], False, True)

		return self.pathToAction(gameState, path)

	def getInvaders(self, gameState):
		invaders = []
		for opponent in self.getOpponents(gameState):
			if self.probOnOurSide(gameState, opponent) > 0.5:
				invaders.append(opponent)

		return invaders

	def chase(self, gameState, invaders):
		ownPos = gameState.getAgentPosition(self.index)
		scared = gameState.getAgentState(self.index).scaredTimer > 0

		invaderToChase = None
		minDist = 999
		for invader in invaders:
			dist = self.getEstimatedDistance(gameState, invader)
			if dist < minDist:
				invaderToChase = invader
				minDist = dist

		invaderPos = gameState.getAgentPosition(invaderToChase)
		if invaderPos and gameState.getAgentState(invaderToChase).isPacman:
			if scared and self.getMazeDistance(ownPos, invaderPos) < 3:
				return self.flee(gameState, invaderPos)
			else:
				return invaderPos

		intersections = self.getMostLikelyIntersections(invaderToChase)
		intersections = list(filter(lambda pos: not self.onOpponentSide(gameState, pos), intersections))

		if intersections:
			bestIntersection = None
			minDist = 999
			for intersection in intersections:
				width = gameState.getWalls().width
				dist = abs(intersection[0] - (width // 2))
				if dist < minDist:
					bestIntersection = intersection
					minDist = dist

			return bestIntersection

		mostLikelyPos = BeliefState.beliefs[invaderToChase].argMax()
		if self.onOpponentSide(gameState, mostLikelyPos):
			width = gameState.getWalls().width
			halfwayPoint = width // 2 - 1 if self.red else width // 2
			if self.red:
				for x in range(mostLikelyPos[0], halfwayPoint):
					if not gameState.hasWall(x, mostLikelyPos[1]):
						mostLikelyPos = (x, mostLikelyPos[1])
			else:
				for x in range(halfwayPoint, mostLikelyPos[0]):
					if not gameState.hasWall(x, mostLikelyPos[1]):
						mostLikelyPos = (x, mostLikelyPos[1])

		return mostLikelyPos

	def flee(self, gameState, pos):
		ownPos = gameState.getAgentPosition(self.index)

		neighbours = getNeighbours(gameState, ownPos, True)

		furthersNeighbour = None
		maxDist = -1
		for neighbour in neighbours:
			dist = self.getMazeDistance(neighbour, pos)
			if dist > maxDist:
				furthersNeighbour = neighbour
				maxDist = dist

		return furthersNeighbour

	def getPatrolPositions(self, gameState):
		foodX, foodY = np.nonzero(np.array(list(self.getFoodYouAreDefending(gameState))))
		foodPositions = np.array(list(zip(foodX, foodY)))

		kMeans = KMeans(n_clusters=2).fit(foodPositions)
		clusterCenters = np.rint(kMeans.cluster_centers_).tolist()

		width = gameState.getWalls().width
		halfwayPoint = width // 2 if self.red else width // 2 + 1

		patrolPositions = []
		for centerX, centerY in clusterCenters:
			closestIntersection = None
			minDist = 999

			for intersection in self.intersections:
				dist = util.manhattanDistance((centerX, centerY), intersection)
				if dist < minDist:
					closestIntersection = intersection
					minDist = dist

			if self.red:
				for x in range(closestIntersection[0], halfwayPoint):
					if not gameState.hasWall(x, closestIntersection[1]):
						closestIntersection = (x, closestIntersection[1])
			else:
				for x in range(halfwayPoint, closestIntersection[0]):
					if not gameState.hasWall(x, closestIntersection[1]):
						closestIntersection = (x, closestIntersection[1])
						break

			patrolPositions.append(closestIntersection)

		return patrolPositions

	def getEstimatedDistance(self, gameState, opponent):
		ownPos = gameState.getAgentPosition(self.index)

		dist = 0
		for pos in BeliefState.beliefs[opponent]:
			prob = BeliefState.beliefs[opponent][pos]

			if prob > 0:
				dist += self.getMazeDistance(ownPos, pos) * prob

		return dist

	def getIntersectionAreas(self, gameState):
		width = gameState.getWalls().width
		height = gameState.getWalls().height

		areas = []
		closedListSuper = []

		for x in range(width):
			for y in range(height):

				if gameState.hasWall(x, y) or (x, y) in self.intersections or (x, y) in closedListSuper:
					continue

				openList = [(x, y)]
				closedList = []
				intersectionList = []

				while openList:
					currentPos = openList[0]

					openList.pop(0)
					closedList.append(currentPos)

					for nodePos in getNeighbours(gameState, currentPos):
						if nodePos in closedList or nodePos in openList:
							continue

						if nodePos in self.intersections:
							intersectionList.append(nodePos)
							continue

						openList.append(nodePos)

				closedListSuper.extend(closedList)
				areas.append([util.Counter(), intersectionList])

				for x2 in range(width):
					for y2 in range(height):
						areas[-1][0][(x2, y2)] = 1 if (x2, y2) in closedList else 0

		return areas

	def getMostLikelyIntersections(self, agent):
		bestArea = None
		bestBelief = -1

		for area in self.areas:
			totalBelief = 0

			for pos in area[0]:
				if BeliefState.beliefs[agent][pos] == 1 and pos in self.intersections:
					return [pos]

				totalBelief += BeliefState.beliefs[agent][pos] * area[0][pos]

			if totalBelief > bestBelief:
				bestArea = area
				bestBelief = totalBelief

		return bestArea[1]

	### DEFENSIVE END ###

	def registerInitialState(self, gameState):
		self.start = gameState.getAgentPosition(self.index)
		self.intersections = getIntersections(gameState)
		self.areas = self.getIntersectionAreas(gameState)

		CaptureAgent.registerInitialState(self, gameState)

		BeliefState.registerInitialState(gameState)

	def chooseAction(self, gameState):

		BeliefState.updateBeliefs(gameState, self)

		self.debugClear()

		# self.displayDistributionsOverPositions(BeliefState.beliefs)

		myTeam = self.getTeam(gameState)
		if self.fullInvasion(gameState) or min(myTeam, default=-1) == self.index:
			return self.defensiveAction(gameState)
		else:
			return self.offensiveAction(gameState)

	### HELPER FUNCTIONS ###

	def fullInvasion(self, gameState, confidence=0.8):
		prob = 1
		for opponent in self.getOpponents(gameState):
			prob *= self.probOnOurSide(gameState, opponent)
		if prob >= confidence:
			return True
		else:
			return False

	def probOnOurSide(self, gameState, opponent):
		prob = 0.0

		for pos in BeliefState.beliefs[opponent]:
			if not self.onOpponentSide(gameState, pos):
				prob += BeliefState.beliefs[opponent][pos]

		return prob

	def pathToAction(self, gameState, path):
		pos = gameState.getAgentPosition(self.index)
		actions = gameState.getLegalActions(self.index)

		if pos in path and pos != path[-1]:
			pathIndex = path.index(pos)

			for action in actions:
				direction = game.Actions.directionToVector(action)
				nextPos = (pos[0] + direction[0], pos[1] + direction[1])
				if nextPos == path[pathIndex + 1]:
					return action

		return actions[-1]

	def aStar(self, gameState, start, end, lowRiskPath=False, forceOwnSide=False):
		startNode = Node(None, start)
		endNode = Node(None, end)

		openList = [startNode]
		closedList = []

		while openList:
			currentNode = openList[0]
			currentIndex = 0
			for index, item in enumerate(openList):
				if item.f < currentNode.f:
					currentNode = item
					currentIndex = index

			openList.pop(currentIndex)
			closedList.append(currentNode)

			if currentNode == endNode:
				path = []
				current = currentNode
				while current:
					path.append(current.pos)
					current = current.parent

				return path[::-1]

			for nodePos in getNeighbours(gameState, currentNode.pos, False):
				if forceOwnSide and self.onOpponentSide(gameState, nodePos):
					continue

				child = Node(currentNode, nodePos)

				if child in closedList:
					continue

				if lowRiskPath and self.onOpponentSide(gameState, nodePos):
					risk = 0
					for opponent in self.getOpponents(gameState):
						if BeliefState.beliefs[opponent][nodePos] > 0.9:
							risk = 9999
							break
						else:
							risk += 10 * BeliefState.beliefs[opponent][nodePos]
					child.g = currentNode.g + 1 + risk
				else:
					child.g = currentNode.g + 1
				child.h = self.getMazeDistance(child.pos, endNode.pos)
				child.f = child.g + child.h

				for openNode in openList:
					if child == openNode and child.g > openNode.g:
						continue

				openList.append(child)

	def onOpponentSide(self, gameState, pos, margin=0):
		width = gameState.getWalls().width
		halfwayPoint = width // 2 - 1 if self.red else width // 2

		if gameState.isOnRedTeam(self.index):
			return pos[0] > halfwayPoint - margin
		else:
			return pos[0] < halfwayPoint + margin

	def debugDisplayPath(self, path, colour=None):
		if colour == None:
			colour = [120, 50, 20] if self.index == 0 else [120, 80, 40]
		self.debugDraw(cells=path, color=colour)


def getPenalty(gameState, pos):
	width = gameState.getWalls().width
	return width // 2 - abs(pos[0] - (width // 2))


def getIntersections(gameState):
	intersections = []
	convexWalls, directions = findConvexWalls(gameState)

	for wallPos, direction in zip(convexWalls, directions):
		if gameState.hasWall(wallPos[0] + direction[0] * 2, wallPos[1] + direction[1] * 2):
			intersectionPos = (wallPos[0] + direction[0], wallPos[1] + direction[1])
			if intersectionPos not in intersections:
				intersections.append(intersectionPos)

	return intersections


def findConvexWalls(gameState):
	convexWalls = []
	directions = []

	for x, col in enumerate(gameState.getWalls()):
		for y in range(len(col)):
			if not gameState.hasWall(x, y):
				continue

			neighbours = getNeighbours(gameState, (x, y), False, True)
			neighbours = list(filter(lambda n: gameState.hasWall(n[0], n[1]), neighbours))

			if len(neighbours) == 1:
				convexWalls.append((x, y))
				directions.append((x - neighbours[0][0], y - neighbours[0][1]))

	return convexWalls, directions


def inBounds(gameState, pos):
	width = gameState.getWalls().width
	height = gameState.getWalls().height

	return 0 <= pos[0] < width and 0 <= pos[1] < height


def getNeighbours(gameState, pos, includePos=False, includeWalls=False):
	neighbours = [pos] if includePos else []
	for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
		neighbour = (pos[0] + direction[0], pos[1] + direction[1])

		if inBounds(gameState, neighbour) and (not gameState.hasWall(neighbour[0], neighbour[1]) or includeWalls):
			neighbours.append(neighbour)

	return neighbours

def nearestPoint( pos ):
	"""
	Finds the nearest grid point to a position (discretizes).
	"""
	( current_row, current_col ) = pos

	grid_row = int( current_row + 0.5 )
	grid_col = int( current_col + 0.5 )
	return ( grid_row, grid_col )


class Node:
	def __init__(self, parent=None, pos=None):
		self.parent = parent
		self.pos = pos

		self.g = 0
		self.h = 0
		self.f = 0

	def __eq__(self, other):
		return self.pos == other.pos
