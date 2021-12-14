#Author: Jonathan Loyd
#Description: Python3 TSP Genetic Algorithm
#CSE545 Project 4

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
def plotRoute(folderName, tour, generation):
    # Variables used to plot the graph
    x = []
    y = []

    # Prepare the graph
    titleString = 'Generation ' + str(generation)
    plt.title(titleString)
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
    plt.savefig(folderName + '\\' + "gen" + str(generation) + '.png')

    # Show the plot to the user
    #plt.show()

    # Clear the plot
    plt.clf()

#*******************************************************
# Plot a graphical of cost versus generation
#*******************************************************
def plotCostVGen(folderName, popList):
    # Variables used to plot the graph
    x = []
    y = []

    # Prepare the graph
    titleString = 'Best Cost Versus Generation'
    plt.title(titleString)
    plt.xlabel('Generation')
    plt.ylabel('Cost')

    # Plot the best cost vs generation
    for pop in popList:
        x.append(pop[1])
        y.append(pop[0])
    xpoints = np.array(x)
    ypoints = np.array(y)
    plt.plot(xpoints,ypoints)

    # Save the figure
    plt.savefig(folderName + '\\' + 'BestCostVsDistance' + '.png')

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
    # Take population limit as the second argument
    popLimit = int(sys.argv[2])
    # Take number of generations as the third argument
    genLimit = int(sys.argv[3])
    # Take percentage of elites as the fourth argument
    percentElites = int(sys.argv[4])
    numElites = int(popLimit * percentElites / 100)
    # Take mutation type as the fifth argument
    mutationType = int(sys.argv[5])
    # Make mutation rate 2%
    mutationRate = float(2)
    mutationRate = mutationRate / 100
    # Take seed value as the 6th argument
    seedValue = int(sys.argv[6])
    # Seed random
    random.seed(seedValue)

    # Create a folder for the run
    folderName = sys.argv[1] + '_' + sys.argv[2] + '_'
    folderName += sys.argv[3] + '_' + sys.argv[4] + '_'
    folderName += sys.argv[5] + '_' + sys.argv[6]
    os.mkdir(folderName)

    # Variable for seeing if ditance improved
    improveDistance = math.inf
    # List of best population from all generations
    popList = list()

    # Initialze total time and start the timer
    totRunTime = 0
    start = time.perf_counter()

    # Get the coordinates in the file
    getCoords(inFilename)

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

    # Stop timer for plotting
    finish = time.perf_counter()
    totRunTime += finish-start
    # Plot the route if there is improvement
    if population[0][1] < improveDistance:
        improveDistance = population[0][1]
        plotRoute(folderName, population[0][0], generation)
    # Start timer again
    start = time.perf_counter()

    # Increment the generation count
    generation += 1

    # Create a new population until the generation limit is hit
    while generation <= genLimit:
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

        # Stop timer for plotting
        finish = time.perf_counter()
        totRunTime += finish-start
        # Plot the route if there is improvement
        if population[0][1] < improveDistance:
            improveDistance = population[0][1]
            plotRoute(folderName, population[0][0], generation)
        # Start timer again
        start = time.perf_counter()

        # Increment the generation count
        generation += 1

    # Stop timer
    finish = time.perf_counter()
    totRunTime += finish-start

    # Plot best cost versus generation
    plotCostVGen(folderName, popList)

    # Append information about the run to a file
    appendInfo(population[0], totRunTime)
