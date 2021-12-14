#Author: Jonathan Loyd
#Description: Python3 TSP Genetic Algorithm w/ Wisdom of Crowds
#CSE545 Project 5

import random, sys, re, math, os, time
import numpy as np
from matplotlib import pyplot as plt

# Global lists of nodes and corresponding coordinates
nodeList = list()
coordsList = list()

# Calculate distance given coordinates of 2 cities
def calcDistance(x, y, a, b):
    distance = math.sqrt(((x-a)**2) + ((y-b)**2))
    return distance

#*******************************************************
# Read in the text file
#*******************************************************
def getCoords(filename):
    infile = open(filename, 'r')
    # Get the coordinates in the tsp file
    coordsTxt = re.search(r'NODE_COORD_SECTION([\s\S]*)', infile.read()).group()
    coordsTxt = coordsTxt.replace('NODE_COORD_SECTION\n', '')
    # Create a list of nodes and coordinates from coordsTxt
    for coord in coordsTxt.splitlines():
        x, y = coord.strip().split()[1:]
        node = int(coord.strip().split()[0])
        # append global lists for later use
        nodeList.append(node)
        coordsList.append([node, float(x), float(y)])

    # Close the tsp file when done using it
    infile.close()

#*******************************************************
# Create an adjacency matrix of all distances for each node to node
#*******************************************************
def getAdjMatrix():
    # Create an empty array of size [i][j] where i and j are the number of nodes
    adjMatrix = np.empty((len(coordsList), len(coordsList)))
    # Fill the array with distances
    for i ,node1 in enumerate(coordsList):
        for j, node2 in enumerate(coordsList):
            adjMatrix[i,j] = calcDistance(node1[1], node1[2], node2[1], node2[2])
    return adjMatrix

#*******************************************************
# Calculate a good fitness measure which will be applied to members of a population
#*******************************************************
def getFitness(distance):
    fitness = 1 / distance
    return fitness

#*******************************************************
# Create an initial population randomly
#*******************************************************
def initPopulation(adjMatrix, popLimit):
    population = list()
    popFitness = 0
    # Create members of the population until its hit the limit
    while len(population) != popLimit:
        newMember = [[], 0, 0]
        # Tour for the member
        newMember[0] = random.sample(nodeList, len(nodeList))
        for count, _ in enumerate(newMember[0][:-1]):
            # The member's corresponding distance
            newMember[1] += adjMatrix[newMember[0][count]-1, newMember[0][count+1]-1]
        # The first and last element for the new member's corresponding distance
        newMember[1] += adjMatrix[newMember[0][0]-1, newMember[0][-1]-1]
        # The member's corresponding fitness
        newMember[2] = getFitness(newMember[1])
        # Add the new member to the population
        population.append(newMember)
        # Add the member's fitness to the total fitness
        popFitness += newMember[2]
    return population, popFitness

#*******************************************************
# Sort the population based on distance
#*******************************************************
def sortPop(population):
    for i in range(1, len(population)):
        index = population[i]

        j = i-1
        while j >= 0 and index[1] < population[j][1]:
            population[j+1] = population[j]
            j -= 1
        population[j+1] = index

#*******************************************************
# Create a likelyhood of the members of the population to reproduce
#*******************************************************
def getReproductionPool(population, totalFitness):
    # Create an empty reproduction pool
    reproductionPool = list()
    # Fill the reproduction pool with member
    for member in population:
        # Populate the reproduction pool based on a members fitness
        # relative to total fitness
        relativeFitness = member[2]/totalFitness
        membersToAdd = round(relativeFitness * len(population) * 100)
        for addedMember in range(membersToAdd):
            reproductionPool.append(member)
    return reproductionPool

#*******************************************************
# Use the concept of elitism to populate some of the new population
#*******************************************************
def getElites(population, newPop, numElites):
    i = 0
    for i in range(numElites):
        newPop.append(population[i])

#*******************************************************
# Reproduce members of the population to create the rest of the new population
# This uses the Partially Mapped Crossover Operator mentioned in this article:
# https://www.hindawi.com/journals/cin/2017/7430125/
#*******************************************************
def reproduce(reproductionPool, newPop, popLimit, mutationRate, mutationType):
    # Find 2 parents from the reproduction pool
    parent1 = reproductionPool[random.randint(0, popLimit-1)]
    parent2 = reproductionPool[random.randint(0, popLimit-1)]

    # Repeatedly find new parents until there are 2 unique parents
    while parent1 == parent2:
        parent1 = reproductionPool[random.randint(0, popLimit-1)]
        parent2 = reproductionPool[random.randint(0, popLimit-1)]

    # Partially create a child
    child = [[], 0, 0]
    child[0] = [0] * len(nodeList)

    # Create cutoffs for what path is kept from the parent
    cutoff1 = random.randint(0, len(nodeList)-1)
    cutoff2 = random.randint(cutoff1, len(nodeList))

    # Variables to help keep track of reproduction
    alreadyInChild = list()
    parentMap = dict()
    zeroIndex = list()

    # Choose a dominant parent at random
    parentSelect = random.random()
    if parentSelect <= .5:
        # Fill part of the child with a section of a parent
        for fill in range(cutoff1, cutoff2):
            child[0][fill] = parent1[0][fill]
            # Keep track of what is already in the child
            alreadyInChild.append(child[0][fill])
            # Map the sharing from the parents for later
            parentMap[parent1[0][fill]] = parent2[0][fill]

        # Fill in as much as possible from the other parent
        for index, _ in enumerate(child[0]):
            if index < cutoff1 or index >= cutoff2:
                if parent2[0][index] not in alreadyInChild:
                    child[0][index] = parent2[0][index]
                else:
                    zeroIndex.append(index)

        # Fill in the remaining nodes based on the parent mapping
        for index in zeroIndex:
            # Attempt to map an input
            attemptInput = parent2[0][index]
            # Find an input that is mapped that isn't already in the child
            while attemptInput in alreadyInChild:
                attemptInput = parentMap[attemptInput]
            # Fill in the allowed input
            child[0][index] = attemptInput

    else:
        # Fill part of the child with a section of a parent
        for fill in range(cutoff1, cutoff2):
            child[0][fill] = parent2[0][fill]
            # Keep track of what is already in the child
            alreadyInChild.append(child[0][fill])
            # Map the sharing from the parents for later
            parentMap[parent2[0][fill]] = parent1[0][fill]

        # Fill in as much as possible from the other parent
        for index, _ in enumerate(child[0]):
            if index < cutoff1 or index >= cutoff2:
                if parent1[0][index] not in alreadyInChild:
                    child[0][index] = parent1[0][index]
                else:
                    zeroIndex.append(index)

        # Fill in the remaining nodes based on the parent mapping
        for index in zeroIndex:
            # Attempt to map an input
            attemptInput = parent1[0][index]
            # Find an input that is mapped that isn't already in the child
            while attemptInput in alreadyInChild:
                attemptInput = parentMap[attemptInput]
            # Fill in the allowed input
            child[0][index] = attemptInput

    # Mutate if the randomly generated number is within the chance of mutation
    mutationCheck = random.random()
    if mutationCheck <= mutationRate:
        mutate(child[0], mutationType)

    # Get the disance of the child
    for count, _ in enumerate(child[0][:-1]):
        # The child's corresponding distance
        child[1] += adjMatrix[child[0][count]-1, child[0][count+1]-1]
    # The first and last element for the child's corresponding distance
    child[1] += adjMatrix[child[0][0]-1, child[0][-1]-1]
    # The child's corresponding fitness
    child[2] = getFitness(child[1])


    # Add the child to the new population
    newPop.append(child)

#*******************************************************
# Based on the mutation rate, potentially mutate the child created from reproduction
#*******************************************************
def mutate(childTour, mutationType):
    if mutationType == 1:
        # This uses Displacement Mutation described here:
        # https://www.researchgate.net/publication/226665831_Genetic_Algorithms_for_the_Travelling_Salesman_Problem_A_Review_of_Representations_and_Operators
        index1 = random.randint(0, len(childTour)-2)
        index2 = random.randint(index1, len(childTour)-1)

        indexDiff = index2 - index1
        moveTour = list()
        for index in range(indexDiff):
            moveTour.append(childTour[index1])
            childTour.remove(childTour[index1])

        index3 = random.randint(0, len(childTour))
        for node in moveTour:
            childTour.insert(index3, node)
    elif mutationType == 2:
        # This uses Exchange Mutation described here:
        # https://www.researchgate.net/publication/226665831_Genetic_Algorithms_for_the_Travelling_Salesman_Problem_A_Review_of_Representations_and_Operators
        index1 = random.randint(0, len(childTour)-1)
        index2 = random.randint(0, len(childTour)-1)

        while index1 == index2:
            index2 = index2 = random.randint(0, len(childTour)-1)

        temp = childTour[index1]
        childTour[index1] = childTour[index2]
        childTour[index2] = temp
    else:
        print("Wrong mutation type.")


#*******************************************************
# Plot a graphical representation of a tour
#*******************************************************
def plotRoute(titleString, tour):
    # Variables used to plot the graph
    x = []
    y = []

    # Prepare the graph
    title = 'Tour: ' + titleString
    plt.title(title)
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # Prepare coordinates of each node in the tour to be plotted
    for node in tour:
        x.append(coordsList[node-1][1])
        y.append(coordsList[node-1][2])
        plt.annotate(node, (coordsList[node-1][1],coordsList[node-1][2]))
    # Add the first node as the ending node
    x.append(coordsList[tour[0]-1][1])
    y.append(coordsList[tour[0]-1][2])
    xpoints = np.array(x)
    ypoints = np.array(y)

    # Plot the coordinates of each node in the tour, and connect them
    plt.plot(xpoints,ypoints, 'o-', markersize = 5,
            markeredgecolor="darkorange", markerfacecolor="darkorange")

    # Prepare coordinates of each node not in the tour to be plotted
    for node in nodeList:
        x.append(coordsList[node-1][1])
        y.append(coordsList[node-1][2])
        plt.annotate(node, (coordsList[node-1][1],coordsList[node-1][2]))
    xpoints = np.array(x)
    ypoints = np.array(y)
    # Plot the coordiates of each node not in the tour,
    # and make sure they are not connected
    plt.scatter(xpoints,ypoints, color="purple")

    # Save the figure
    plt.savefig(titleString + '.png')

    # Show the plot to the user
    #plt.show()

    # Clear the plot
    plt.clf()

#*******************************************************
# Append info to a file
#*******************************************************
def appendInfo(bestMember, totRunTime):
    outFile = open('run_info.txt', 'a')
    # Append the arguments for the run
    writeString = 'Args: ' + str(sys.argv) + '\n'
    outFile.write(writeString)
    # Append the best tour and its cost
    writeString = 'Best tour: ' + str(bestMember[0]) + '\n'
    writeString += 'Cost: ' + str(bestMember[1]) + '\n'
    outFile.write(writeString)
    # Append the run time
    writeString = 'Run time: ' + str(totRunTime) + ' seconds\n\n'
    outFile.write(writeString)

# Main function
if __name__ == "__main__":
    # Take file input as the first argument when running program
    inFilename = str(sys.argv[1])
    # Take seed value as the 2nd argument
    seedValue = int(sys.argv[2])
    # Seed random
    random.seed(seedValue)

    totalGAtime = 0
    totalWOCtime = 0
    # TESTING ONE RUN
    outFile = open('run_info.txt', 'a')
    # TESTING MULTIPLE QUICK RUNS
    # outFile = open('bad_input_run_info.txt', 'a')
    writeString = str(inFilename) + " " + str(seedValue)
    outFile.write(writeString)

    # Get the coordinates in the file
    getCoords(inFilename)

    # Create a blank agreementMatrix and fill it with 0s (WOC)
    agreementMatrix = np.empty((len(coordsList), len(coordsList)))
    agreementMatrix.fill(0)

    bestPath = [0, math.inf, 0]

    # TESTING SINGLE RUN
    for _ in range(1):
    # TESTING MULTIPLE QUICK RUNS
    # for _ in range(25):
        GAstart = time.perf_counter()
        # Take population limit as the second argument
        #popLimit = random.randint(500,2000)

        # TESTING MULTIPLE QUICK RUNS
        # popLimit = 2 * len(coordsList)
        # TESTING SINGLE RUN
        popLimit = 10 * (len(coordsList))


        # # Take number of generations as the third argument
        # genLimit = random.randint(50,500)
        # Take percentage of elites as the fourth argument
        percentElites = 10
        numElites = int(popLimit * percentElites / 100)
        # Take mutation type as the fifth argument
        # TESTING SINGLE RUN
        mutationType = 1
        # TESTING MULTIPLE QUICK RUNS
        # mutationType = random.randint(1,2)
        # TESTING SINGLE RUN
        mutationRate = float(5)
        # TESTING MULTIPLE QUICK RUNS
        # mutationRate = float(random.randint(1,5))
        mutationRate = mutationRate / 100

        # Variable for seeing if distance improved
        improveDistance = math.inf
        gensNoImprove = 0
        # List of best population from all generations
        popList = list()

        # Create an adjacency matrix for the distance
        adjMatrix = getAdjMatrix()

        # Initialize a generation count
        generation = 0

        # Create an initial population
        population, totalFitness = initPopulation(adjMatrix, popLimit)
        # Sort the population from shortest to largest distance
        sortPop(population)
        # Print the best distance from that generation
        print("Gen", generation, "distance:", population[0][1])
        # Add the best member distance to the list
        popList.append([population[0][1], generation])

        # Increment the generation count
        generation += 1

        # Get connections of nodes for members in a population (WOC)
        for member in population[0:numElites-1]:
            # Get node to node connections (except for last to first)
            for i, node in enumerate(member[0][:-1]):
                agreementMatrix[(member[0][i]-1), (member[0][i+1]-1)] += (1/adjMatrix[(member[0][i]-1), (member[0][i+1]-1)])
            # Get connection from last node to first node
            agreementMatrix[(member[0][-1]-1), (member[0][0]-1)] += (1/adjMatrix[(member[0][-1]-1), (member[0][0]-1)])
        #print(agreementMatrix)


        #while population[0][0] != population[numElites-1][0]:

        # TESTING SINGLE RUN
        stopGens = 15 * (len(coordsList))

        # TESTING MULTIPLE QUICK RUNS
        # stopGens = int(random.randint(1,5) * (len(coordsList))/2)
        while gensNoImprove <= stopGens:

        # Create a new population until the generation limit is hit
        #while generation <= genLimit:
            # Create an empty new population
            newPop = list()

            # Get the initial reproduction pool
            reproductionPool = getReproductionPool(population, totalFitness)
            # Get and put elites into the new population
            getElites(population, newPop, numElites)

            # Reproduce members of the population until it hits the population limit
            while len(newPop) < popLimit:
                reproduce(population, newPop, popLimit, mutationRate, mutationType)

            # Make the current population the newly created population
            population = newPop
            # Sort the population from shortest to largest distance
            sortPop(population)
            # Print the best distance from that generation
            print("Gen", generation, "distance:", population[0][1])
            # Add the best member distance to the list
            popList.append([population[0][1], generation])

            # Increment the generation count
            generation += 1

            gensNoImprove += 1

            if population[0][1] < improveDistance:
                improveDistance = population[0][1]
                gensNoImprove = 0

        if population[0][1] < bestPath[1]:
            bestPath = population[0]

        GAfinish = time.perf_counter()
        totalGAtime += GAfinish - GAstart
        # # Plot the route
        # plotRoute("PopMember" + str(count), member[0])

        WOCstart = time.perf_counter()
        # Get connections of nodes for members in a population (WOC)
        for count, member in enumerate(population[0:numElites-1]):
        #for member in population[0:2]:
            # # Plot the route
            # plotRoute("PopMember" + str(count), member[0])
            # Get node to node connections (except for last to first)
            for i, node in enumerate(member[0][:-1]):
                agreementMatrix[(member[0][i]-1), (member[0][i+1]-1)] += (1/adjMatrix[(member[0][i]-1), (member[0][i+1]-1)])
            # Get connection from last node to first node
            agreementMatrix[(member[0][-1]-1), (member[0][0]-1)] += (1/adjMatrix[(member[0][-1]-1), (member[0][0]-1)])
        #print(agreementMatrix)
        WOCfinish = time.perf_counter()
        totalWOCtime += WOCfinish - WOCstart

    print("\nGA:")
    print("Best Path:", bestPath[0])
    print("Distance:", bestPath[1])
    print("Time:", totalGAtime)
    writeString = "\nGA:" + "\nBest Path: " + str(bestPath[0]) + "\nDistance: " + str(bestPath[1]) + "\nTime: " + str(totalGAtime)
    outFile.write(writeString)

    # Plot the best GA
    GAstring = "GA" + "_" + str(seedValue) + "_" + str(len(coordsList))
    plotRoute(GAstring, bestPath[0])


    WOCstart = time.perf_counter()
    # Combine the agreementMatrix values, convert the matrix to a cost matrix, and mirror the matrix (WOC)
    for i, node1 in enumerate(coordsList):
        for j, node2 in enumerate(coordsList):
            if j > i:
                # Combine the values
                agreementMatrix[i,j] += agreementMatrix[j,i]
                # Convert to cost matrix
                agreementMatrix[i,j] = 1 - agreementMatrix[i,j]
                # Mirror the graph
                agreementMatrix[j,i] = agreementMatrix[i,j]
            elif i == j:
                agreementMatrix[i,j] = 1
    print(agreementMatrix)

    # Find the global minimum of the agreement matrix (which is now the cost matrix)
    wocPath = list()
    minimum = math.inf
    for m in range(len(coordsList)):
        for n in range(len(coordsList)):
            if agreementMatrix[m,n] < minimum:
                minimum = agreementMatrix[m,n]
                index1 = m+1
                index2 = n+1
    wocPath.append(index1)
    wocPath.append(index2)

    # Complete the rest of the WOC path based on the agreement matrix
    while(len(wocPath) < len(coordsList)):
        m = wocPath[-1]-1
        minimum = math.inf
        for n in range(len(coordsList)):
            if (n+1 not in wocPath) and (agreementMatrix[m,n] < minimum):
                minimum = agreementMatrix[m,n]
                index2 = n+1
        wocPath.append(index2)



    # Calulate the WOC path distance
    wocDistance = 0
    for count, _ in enumerate(wocPath[:-1]):
        # The wocPath corresponding distance
        wocDistance += adjMatrix[wocPath[count]-1, wocPath[count+1]-1]
    # The first and last element for the wocPath's corresponding distance
    wocDistance += adjMatrix[wocPath[-1]-1, wocPath[0]-1]

    WOCfinish = time.perf_counter()
    totalWOCtime += WOCfinish - WOCstart

    # Print out the path
    print("\nWOC:")
    print("Path:", wocPath)
    print("Distance:", wocDistance)
    print("Time:", totalWOCtime)
    writeString = "\nWOC:" + "\nPath: " + str(wocPath) + "\nDistance: " + str(wocDistance) + "\nTime: " + str(totalWOCtime) + "\n\n"
    outFile.write(writeString)


    # Plot the WOC path
    WOCstring = "WOC" + "_" + str(seedValue) + "_" + str(len(coordsList))
    plotRoute(WOCstring, wocPath)

'''
CHECKLIST
Create a population using some different variations of the genetic algorithm


Create a blank agreement matrix

Go through every member of the population and get what each node is connected to
    Add that connection to the agreement matrix at spot Aij where i and j are the connected nodes

Turn that agreement into a cost matrix by doing Cij = 1-Aij


We can then solve for the minimum spanning tree (MSTP) over the cost matrix by obtaining the spanning tree that maximizes
the agreement with solutions in the populations
This solving can be done with Prim's algorithm

'''
