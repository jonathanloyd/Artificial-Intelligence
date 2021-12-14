#Author: Jonathan Loyd
#Description: Python3 TSP Brute Force
#CSE545 Project 1

# import libraries
import re, math, time, sys
from itertools import permutations
import numpy as np
from matplotlib import pyplot as plt

# global lists of nodes and corresponding coordinates
nodeList = []
coordsList = []
# a global list for distances and corresponding routes
distanceList = []

# calculate distance given coordinates of 2 cities
def calcDistance(x, y, a, b):
    distance = math.sqrt(((x-a)**2) + ((y-b)**2))
    return distance

# get coordinates from a tsp file and put it in 2 global lists
def getCoords(filename):
    #filename = input("Please enter the name of the file: ")
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

# calculate least cost route through brute force
def tspBruteForce():
    # get all possible permutations
    perm = permutations(nodeList[1:])
    # iterate through all permutations
    for permut in list(perm):
        # first city constant, so all routes begin with 1
        route = ['1']
        # clear distance variable
        distance = 0
        # calculate distance from first node to permutation node
        distance += calcDistance(coordsList[0][1],
                                    coordsList[0][2],
                                    coordsList[int(permut[0])-1][1],
                                    coordsList[int(permut[0])-1][2])
        # go through a tuple of nodes in each permutation
        for i, node in enumerate(permut):
            route.append(node)
            # helps find node in coordsList
            thisNode = int(permut[i])-1
            # if hit end of permutations, next node is starting city
            if i == len(permut)-1:
                nextNode = 0
            # otherwise next node is next in permutation tuple
            else:
                nextNode = int(permut[i+1])-1
            # calculate distance from node to node
            distance += calcDistance(coordsList[thisNode][1],
                                        coordsList[thisNode][2],
                                        coordsList[nextNode][1],
                                        coordsList[nextNode][2])
        # end at the city you started which is constantly 1 in this case
        route.append('1')
        # append list for distances and corresponding routes
        distanceList.append([distance, route])

# print the minimum distance in the list and its corresponding route
def printRoute(filename):
    print('Optimal Route and Cost for', filename)
    print('Route: ', min(distanceList[0:])[1])
    print('Cost: ', min(distanceList[0:])[0])

# graphical representation of path traveled
def plotRoute(filename):
    x = []
    y = []
    for node in min(distanceList[0:])[1]:
        x.append(coordsList[int(node)-1][1])
        y.append(coordsList[int(node)-1][2])
    titleString = 'Optimal Route for ' + filename
    plt.title(titleString)
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.plot(x,y)
    plt.show()

# main function
if __name__ == "__main__":
    #take file input as the first argument when running program
    filename = str(sys.argv[1])
    getCoords(filename)
    start = time.time()
    tspBruteForce()
    printRoute(filename)
    end = time.time()
    print("Time Elapsed: ", end-start, "seconds")
    plotRoute(filename)
