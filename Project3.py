#Author: Jonathan Loyd
#Description: Python3 TSP Greedy Edge Algorithm
#CSE545 Project 3

import re, math, time, sys
from itertools import permutations
import numpy as np
from matplotlib import pyplot as plt

# global lists of nodes and corresponding coordinates
nodeList = []
coordsList = []

# global variable to help save files
pltIt = 0

# calculate distance given coordinates of 2 cities
def calcDistance(x, y, a, b):
    distance = math.sqrt(((x-a)**2) + ((y-b)**2))
    return distance

# calculate the gamma value using law of cosines
def calcGamma(ls, ss, ab):
    top = ((ss ** 2)  + (ab ** 2) - (ls ** 2))
    bottom = (2 * ab * ss)
    gamma = math.acos((top / bottom))
    return gamma

def calcEdgeDistance(a, b, c):
    # distance for existing nodes
    ab = calcDistance(coordsList[a-1][1],
                    coordsList[a-1][2],
                    coordsList[b-1][1],
                    coordsList[b-1][2])
    # distance between existing node and possible new node
    ac = calcDistance(coordsList[a-1][1],
                    coordsList[a-1][2],
                    coordsList[c-1][1],
                    coordsList[c-1][2])
    # distance between other existing node and possible new node
    bc = calcDistance(coordsList[b-1][1],
                    coordsList[b-1][2],
                    coordsList[c-1][1],
                    coordsList[c-1][2])

    # check which side is shorter and longer
    if ac >= bc:
        ss = bc
        ls = ac
    else:
        ss = ac
        ls = bc

    # calulate the gamma value (an angle value)
    gamma = calcGamma(ls, ss, ab)

    # if gamma angle > 90 degrees (pi/2), return shortest side
    if gamma > ((math.pi) / 2):
        return ss
    # otherwise, calculate the height from the line to the node
    # and return the height
    else:
        s = (ab + ac + bc) / 2
        areaTri = math.sqrt(s * (s - ab) * (s - bc) * (s - ac))
        height = ((2 * areaTri) / ab)
        return height

# get coordinates from a tsp file and put it in 2 global lists
def getCoords(filename):
    infile = open(filename, 'r')
    # get the coordinates in the tsp file
    coordsTxt = re.search(r'NODE_COORD_SECTION([\s\S]*)', infile.read()).group()
    coordsTxt = coordsTxt.replace('NODE_COORD_SECTION\n', '')
    # create a list of nodes and coordinates from coordsTxt
    for coord in coordsTxt.splitlines():
        x, y = coord.strip().split()[1:]
        node = int(coord.strip().split()[0])
        # append global lists for later use
        nodeList.append(node)
        coordsList.append([node, float(x), float(y)])

    # close the tsp file when done using it
    infile.close()


def pickStartNodes(tourList):
    # solve trivially if less than 4 nodes
    if ((len(nodeList) <= 3) and (len(nodeList) >= 0)):
        for node in nodeList:
            tourList.append(node)

    # else find optimal starting nodes for greedy insertion algorithm
    else:
        permList = list()
        # get all permutations of length 2 in nodeList
        perm = permutations(nodeList, 2)
        # add in any permutation that is not the reverse of a
        # permutation already added
        for p in perm:
            if p <= p[::-1]:
                permList.append(p)

        # variables used to check for the best permutation of starting nodes
        # permDistance value can be changed to math.inf to help find nodes
        # with shortest distance instead of farthest
        permDistance = 0
        bestPermutation = []

        # check permutations for the distance between nodes in them
        for perm in permList:
            d1 = calcDistance(coordsList[perm[0]-1][1],
                            coordsList[perm[0]-1][2],
                            coordsList[perm[1]-1][1],
                            coordsList[perm[1]-1][2])

            # check if the combined distance is more than the longest
            # permutation distance already calculated
            # change the sign from > to < to find shortest distance
            # starting nodes (performs worse)
            if d1 > permDistance:
                permDistance = d1
                bestPermutation = perm

        # add nodes in the best permutation to the tourList and remove
        # from nodeList
        for node in bestPermutation:
            tourList.append(node)
            nodeList.remove(node)
            #print(node)
        # add the start node to the end of the tour
        tourList.append(bestPermutation[0])
        plotRoute(tourList)


'''# nearest insertion algorithm, that inserts the closest node to a node already
# in the tour until all nodes have been visited
def greedyNodeTour(tourList):
    while nodeList != []:
        # used to keep track of the node with the shortest distance to it
        routeDist = math.inf
        routeNode = 0
        index = 0

        # iterate through tourList size-1
        # this will check each edge and we check each set of nodes
        # for the closest node then we add the overall closest node
        # to insert into the tour and remove the node from nodeList
        for count, _ in enumerate(tourList[:-1]):
            # get the distance between a node and its connected node in the tour
            existDist = calcDistance(coordsList[tourList[count]-1][1],
                                    coordsList[tourList[count]-1][2],
                                    coordsList[tourList[count+1]-1][1],
                                    coordsList[tourList[count+1]-1][2])
            # check for the closest node remaining to add to the tour
            for node2 in nodeList:
                # get the distance between a node in nodeList and the edge nodes
                # in the tourList
                edgeDist1 = calcDistance(coordsList[tourList[count]-1][1],
                                        coordsList[tourList[count]-1][2],
                                        coordsList[node2-1][1],
                                        coordsList[node2-1][2])
                edgeDist2 = calcDistance(coordsList[tourList[count+1]-1][1],
                                        coordsList[tourList[count+1]-1][2],
                                        coordsList[node2-1][1],
                                        coordsList[node2-1][2])
                # add distances to the edge nodes, and subtract the distance between
                # the connected nodes already in the tour
                totalDistance = (edgeDist1 + edgeDist2) - existDist
                # check if this total distance is less than
                # the current routeDist
                if totalDistance < routeDist:
                    # if yes, then replace routeDist with totalDistance
                    # and replace routeNode with the newest closest node
                    routeDist = totalDistance
                    routeNode = node2
                    index = count+1
                    #print(tourList[count], tourList[count+1], node2)
        # after all nodes have been checked, insert the closest to tourList
        tourList.insert(index, routeNode)
        # and remove the closest node from the list of
        # remaining nodes to be added
        nodeList.remove(routeNode)'''

def greedyEdgeTour(tourList):
    while nodeList != []:
        # used to keep track of the node with the shortest distance to it
        routeDist = math.inf
        routeNode = 0
        index = 0
        baseNodes = []
        # check for closest node to any given edge in the tour
        for count, _ in enumerate(tourList[:-1]):
            for node2 in nodeList:
                # calculate the distance from the edge to the node
                totalDistance = calcEdgeDistance(tourList[count],
                                                tourList[count+1],
                                                node2)
                if totalDistance < routeDist:
                    # if above is true, then replace routeDist with totalDistance
                    # and replace routeNode with the newest closest node
                    routeDist = totalDistance
                    routeNode = node2
                    index = count+1
        # after all nodes have been checked, insert the closest to tourList
        tourList.insert(index, routeNode)
        # and remove the closest node from the list of
        # remaining nodes to be added
        nodeList.remove(routeNode)
        #print(tourList)
        plotRoute(tourList)
        #print(routeNode)

# graphical representation of path traveled
def plotRoute(tourList):
    # variables used to plot the graph
    x = []
    y = []

    # prepare the graph
    titleString = 'The Tour'
    plt.title(titleString)
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # prepare coordinates of each node in the tour to be plotted
    for node in tourList:
        #print(coordsList[node-1])
        x.append(coordsList[node-1][1])
        y.append(coordsList[node-1][2])
        plt.annotate(node, (coordsList[node-1][1],coordsList[node-1][2]))
    xpoints = np.array(x)
    ypoints = np.array(y)

    # plot the coordinates of each node in the tour, and connect them
    plt.plot(xpoints,ypoints, 'o-', markersize = 5,
            markeredgecolor="darkorange", markerfacecolor="darkorange")

    # prepare coordinates of each node not in the tour to be plotted
    for node in nodeList:
        x.append(coordsList[node-1][1])
        y.append(coordsList[node-1][2])
        plt.annotate(node, (coordsList[node-1][1],coordsList[node-1][2]))
    xpoints = np.array(x)
    ypoints = np.array(y)
    # plot the coordiates of each node not in the tour,
    # and make sure they are not connected
    plt.scatter(xpoints,ypoints, color="purple")

    # global variable to help save files
    global pltIt
    pltIt += 1

    # save the figure
    plt.savefig("file" + str(pltIt) + '.png')

    # show the plot to the user
    #plt.show()

    #clear the plot
    plt.clf()


# main function
if __name__ == "__main__":
    # take file input as the first argument when running program
    filename = str(sys.argv[1])
    # get the coordinates in the file
    getCoords(filename)

    # conduct the greedy tour
    # if there are no nodes in a path
    if len(nodeList) == 0:
        print("There are no nodes to find a path for\n")

    # solve if there are nodes in a path
    else:
        # initialize an empty tour list
        tourList = list()

        start = time.perf_counter()
        # solve trivially for <4 nodes and get the starting nodes for >4 nodes
        pickStartNodes(tourList)
        print(tourList)

        # if >4 nodes, then conduct nearest insertion greedy algorithm
        # to insert the remaining nodes
        greedyEdgeTour(tourList)
        print(tourList)

        # get and print the distance for the tour
        dist2 = 0
        for count, _ in enumerate(tourList[:-1]):
            dist2 += calcDistance(coordsList[tourList[count]-1][1],
                                    coordsList[tourList[count]-1][2],
                                    coordsList[tourList[count+1]-1][1],
                                    coordsList[tourList[count+1]-1][2])
        print(dist2)
        end = time.perf_counter()
        print("Time taken: ", end-start)
