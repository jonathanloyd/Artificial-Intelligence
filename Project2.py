#Author: Jonathan Loyd
#Description: Python3 TSP BFS vs. DFS
#CSE545 Project 2

import re, math, time, sys
from collections import defaultdict

# global lists of nodes and corresponding coordinates
nodeList = []
coordsList = []

# calculate distance given coordinates of 2 cities
def calcDistance(x, y, a, b):
    distance = math.sqrt(((x-a)**2) + ((y-b)**2))
    return distance

# get coordinates from a tsp file and put it in 2 global lists
def getCoords(filename):
    infile = open(filename, 'r')
    # get the coordinates in the tsp file
    coordsTxt = re.search(r'NODE_COORD_SECTION([\s\S]*)', infile.read()).group()
    coordsTxt = coordsTxt.replace('NODE_COORD_SECTION\n', '')
    # create a list of nodes and coordinates from coordsTxt
    for coord in coordsTxt.splitlines():
        x, y = coord.strip().split()[1:]
        node = coord.strip().split()[0]
        # append global lists for later use
        nodeList.append(node)
        coordsList.append([node, float(x), float(y)])
    # close the tsp file when done using it
    infile.close()

# conduct BFS
def breadthFirstSearch(routeDict, sourceNode, endNode):
    # create am empty queue and push a distance of 0 into it
    # corresponding with the first path
    queue2 = []
    startDist = 0
    queue2.append([startDist])

    # create an empty queue and push the first path into it
    # which is just the source node
    queue = []
    queue.append([sourceNode])
    # go through the queue
    while queue:
        # pop the path out of the queue
        path = queue.pop(0)
        # set recVisNode equal to the most recently visted node in the path
        recVisNode = path[-1]
        # pop the distance associated with path out of the queue
        distForPath = queue2.pop(0)
        # set recDist equal to the most recently visited path distance
        recDist = distForPath[-1]
        # if the end node is found, then the path
        # goes from source node to end node
        # and we check it against the minimum distance path
        if recVisNode == endNode:
            # if its less than minimum distance path, its the new minimum
            # distance path
            if recDist < bfsMinDist[0]:
                bfsMinDist[0] = recDist
                bfsMinDist[1] = path
        # visit all adjacent nodes then create a path
        # and push it to the queue
        for adjacentNode in routeDict.get(recVisNode, []):
            # keep track of path
            createdPath = list(path)
            createdPath.append(adjacentNode)
            queue.append(createdPath)

            # calculate and keep track of the path's associated distance
            distance = recDist + calcDistance(coordsList[createdPath[-2]-1][1],
                                        coordsList[createdPath[-2]-1][2],
                                        coordsList[createdPath[-1]-1][1],
                                        coordsList[createdPath[-1]-1][2])
            listForDist = [distance]
            newDist = list(distForPath)
            queue2.append(listForDist)
    print("Route: ", bfsMinDist[1], "Distance: ", bfsMinDist[0])

def dfsRecursive(node, path, totDistance):
    # explore all adjacent nodes of the node called
    for adjacentNode in routeDict[node]:
        # calculate distance from the called node to the adjacent node
        distance = calcDistance(coordsList[node-1][1],
                                coordsList[node-1][2],
                                coordsList[adjacentNode-1][1],
                                coordsList[adjacentNode-1][2])
        # increase total distance of path by distance
        totDistance += distance
        # append the adjacent node to the current path
        path.append(adjacentNode)
        # if the path has reached the end node then check if the distance
        # is less than the previously stored distance
        if adjacentNode == 11:
            if totDistance < dfsMinDist[0]:
                dfsMinDist[0] = totDistance
                dfsMinDist[1] = path.copy()
        # call DFS recursively for all adjacent nodes
        dfsRecursive(adjacentNode, path, totDistance)
        # pop the path when going back up the graph
        path.pop()
        # removed distance traveled when going back up the graph
        totDistance -= distance

def depthFirstSearch(routeDict, sourceNode):
    # variables for keeping track of path and total distance of a path
    path = [sourceNode]
    totDistance = 0
    # call DFS recursively
    dfsRecursive(sourceNode, path, totDistance)
    # print out the minimum distance route, and its corresponding distance
    print("Route: ", dfsMinDist[1], "Distance: ", dfsMinDist[0])

# main function
if __name__ == "__main__":
    # take file input as the first argument when running program
    filename = str(sys.argv[1])
    # get the coordinates in the file
    getCoords(filename)

    # dictionary of possible routes
    routeDict = {
                    1: [2, 3, 4],
                    2: [3],
                    3: [4, 5],
                    4: [5, 6, 7],
                    5: [7, 8],
                    6: [8],
                    7: [9, 10],
                    8: [9, 10, 11],
                    9: [11],
                    10: [11]
                }
    # make a list to keep track of the minimum distance for BFS
    bfsMinDist = [math.inf, []]

    start = time.perf_counter()
    # BFS function
    # with 1 as the source node
    breadthFirstSearch(routeDict, 1, 11)
    end = time.perf_counter()
    print('Time Elapsed: ', end-start, 'seconds')

    # add a path from 11 to itself for DFS to work properly
    routeDict[11] = []
    # make a list to keep track of the minimum distance for DFS
    dfsMinDist = [math.inf, []]

    start = time.perf_counter()
    # DFS function
    # with 1 as the source node
    depthFirstSearch(routeDict, 1)
    end = time.perf_counter()
    print('Time Elapsed: ', end-start, 'seconds')
