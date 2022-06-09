
# pathCalculator.py
# Herman Blenneros (hermanbl@kth.se)
# ---------------------
# Heavily inspired by
# distanceCalculator.py
# ---------------------
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


"""
This file contains a Pather object which computes the shortest path between any two points in the maze.

Example:
pather = Pather(gameState.data.layout)
pather.getPath( (1,1), (10,10) )
"""

import sys, time, random
import util

class Pather:
    def __init__(self, layout, default = 10000):
        self.layout = layout
        self.default = default

    def getPath(self, pos1, pos2, enemy1 = None, enemy2 = None):

        allNodes = self.layout.walls.asList(False)
        safeNodes = allNodes.copy()

        if enemy1 is not None:
            enemy_1x = int(enemy1[0])
            enemy_1y = int(enemy1[1])

            dilate_enemy1 = [(enemy_1x, enemy_1y),
                            (enemy_1x + 1, enemy_1y),
                            (enemy_1x - 1, enemy_1y),
                            (enemy_1x, enemy_1y + 1),
                            (enemy_1x, enemy_1y - 1)]

            for enemy in dilate_enemy1:
                if enemy in safeNodes:
                    safeNodes.remove(enemy)
        
        if enemy2 is not None:
            enemy_2x = int(enemy2[0])
            enemy_2y = int(enemy2[1])

            dilate_enemy2 = [(enemy_2x, enemy_2y),
                            (enemy_2x + 1, enemy_2y),
                            (enemy_2x - 1, enemy_2y),
                            (enemy_2x, enemy_2y + 1),
                            (enemy_2x, enemy_2y - 1)]

            for enemy in dilate_enemy2:
                if enemy in safeNodes:
                    safeNodes.remove(enemy)
        
        try: 
            safeNodes.index(pos1)
        except: return None

        try: 
            safeNodes.index(pos2)
        except: return None

        path = computePath(self.layout, safeNodes, pos1, pos2)

        return path

    def getPath_withoutlist(self, pos1, pos2, enemyList = None):

        allNodes = self.layout.walls.asList(False)
        safeNodes = allNodes.copy()

        if enemyList is not None:
            for enemy in enemyList:
                if enemy in safeNodes:
                    safeNodes.remove(enemy)
        
        try: 
            safeNodes.index(pos1)
        except: return None

        try: 
            safeNodes.index(pos2)
        except: return None

        path = computePath(self.layout, safeNodes, pos1, pos2)

        return path

def computePath(layout, safeNodes, pos1, pos2):
    """
    Finds the shortest path between pos1 and pos2, in that order, using uniform cost search
    """
    cost = {}
    parent = {}
    closed = {}
    
    for node in safeNodes:
        cost[node] = sys.maxsize
    
    queue = util.PriorityQueue()
    queue.push(pos1,0)
    cost[pos1] = 0
    parent[pos1] = -1
    while not queue.isEmpty():
        node = queue.pop()

        if node == pos2:
            path = unravelPath(pos1, pos2, parent)
            return path

        if node in closed:
            continue
        closed[node] = True
        nodeCost = cost[node]
        adjacent = []
        # x, y = node
        x = int(node[0])
        y = int(node[1])
        if not layout.isWall((x,y+1)):
            adjacent.append((x,y+1))
        if not layout.isWall((x,y-1)):
            adjacent.append((x,y-1) )
        if not layout.isWall((x+1,y)):
            adjacent.append((x+1,y) )
        if not layout.isWall((x-1,y)):
            adjacent.append((x-1,y))
        for other in adjacent:
            # Safety measure
            if not other in cost:
                continue
            oldCost = cost[other]
            newCost = nodeCost + 1
            if newCost < oldCost:
                cost[other] = newCost
                parent[other] = node
                queue.push(other, newCost)
    return None

global path

def unravelPath(pos1, pos2, parent):

    node = pos2
    path = []

    while parent[node] != -1:
        path.insert(0, node)
        node = parent[node]
    
    path.insert(0, node)

    return path
