#Authors: Jonathan Loyd, Caty Battjes, Nicole Hartman
#Description: Python3 Number Partitioning Genetic Algorithm w/ Wisdom of Crowds
#CSE545 Project 6

import sys, random, math, copy
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

# Read the given data file
def read_data(filename):
    infile = open(filename, 'r')
    for line in infile:
        number_set.append(int(line))
    infile.close()

# Initialize a population for GA
def init_population(population_limit):
    population = list()
    population_fitness = 0
    while len(population) != population_limit:
        set1 = list()
        set1_sum = 0
        set2 = list()
        set2_sum = 0
        new_member_fitness = 0

        for number in number_set:
            if (random.randint(1, 2) == 1):
                set1.append(number)
                set1_sum += number
            else:
                set2.append(number)
                set2_sum += number
        new_member_fitness = 1 / (abs(set1_sum - set2_sum) + 1)
        new_member = [set1, set2, new_member_fitness, set1_sum, set2_sum]
        population.append(new_member)
        population_fitness += new_member_fitness
    return population, population_fitness

def sort_population(population):
    for i in range (1, len(population)):
        index = population[i]
        j = i-1
        while j >= 0 and index[2] > population[j][2]:
            population[j+1] = population[j]
            j -=1
        population[j+1] = index

def get_elites(population, new_population, number_elites, new_population_fitness):
    i = 0
    for i in range(number_elites):
        new_population.append(population[i])
        new_population_fitness += population[i][2]

def get_reproduction_pool(population, population_fitness):
    reproduction_pool = list()
    for member in population:
        relative_fitness = member[2]/population_fitness
        members_to_add = round(relative_fitness * len(population) * 100)
        for added_member in range(members_to_add):
            reproduction_pool.append(member)
    return reproduction_pool

def mutate(child, mutation_rate, mutation_happen):
    flag = 0

    # Pick a set from the child that is not empty
    # Move a random number of numbers in that set into the other set
    # Also adjust the fitness of the child to reflect the swap
    if (random.randint(1, 2) == 1):
        select_set = child[0]
        flag = 1
    else:
        select_set = child[1]
        flag = 2
    if (len(select_set) == 0 and flag == 1):
        select_set = child[1]
        flag = 2
    elif(len(select_set) == 0 and flag == 2):
        select_set = child[0]
        flag = 1
    else:
        pass
    index = 0
    number_to_swap = 1
    # Small chance to move 1 or more numbers into the other set
    if mutation_happen <= (mutation_rate/5):
        if len(select_set) > 1:
            number_to_swap = random.randint(1, len(select_set))
        for i in range(number_to_swap):
            index = random.randint(0, len(select_set)-1)
            if (flag == 1):
                child[3] -= child[0][index]
                child[4] += child[0][index]
                child[1].append(child[0][index])
                child[0].pop(index)
            else:
                child[3] += child[1][index]
                child[4] -= child[1][index]
                child[0].append(child[1][index])
                child[1].pop(index)
    # Bigger chance to move only 1 number from a set to another
    else:
        if len(select_set) > 1:
            index = random.randint(0, len(select_set)-1)
        if (flag == 1):
            child[3] -= child[0][index]
            child[4] += child[0][index]
            child[1].append(child[0][index])
            child[0].pop(index)
        else:
            child[3] += child[1][index]
            child[4] -= child[1][index]
            child[0].append(child[1][index])
            child[1].pop(index)

    # Adjust the fitness of the child
    child[2] = 1 / (abs(child[3] - child[4]) + 1)

def reproduce(population, new_population, reproduction_pool, mutation_rate, new_population_fitness):
    parent1 = reproduction_pool[random.randint(0, len(reproduction_pool)-1)]
    parent2 = reproduction_pool[random.randint(0, len(reproduction_pool)-1)]

    # Create a duplicate (deepcopy) of parent2 because we adjust the list itself
    temp_parent2 = copy.deepcopy(parent2)
    child_set1 = list()
    child_set1_sum = 0
    child_set2 = list()
    child_set2_sum = 0
    child_fitness = 0

    # Get the elements that are the same in s1 in both parents and then s2 in both parents
    # If not, randomly place it in s1 or s2 for the child
    for number in parent1[0]:
        if number in temp_parent2[0]:
            child_set1.append(number)
            temp_parent2[0].remove(number)
            child_set1_sum += number
        else:
            if (random.randint(1,2) == 1):
                child_set1.append(number)
                child_set1_sum += number
            else:
                child_set2.append(number)
                child_set2_sum += number
    for number in parent1[1]:
        if number in temp_parent2[1]:
            child_set2.append(number)
            temp_parent2[1].remove(number)
            child_set2_sum += number
        else:
            if (random.randint(1,2) == 1):
                child_set1.append(number)
                child_set1_sum += number
            else:
                child_set2.append(number)
                child_set2_sum += number

    # Adjust the fitness of the child and create it
    child_fitness = 1 / (abs(child_set1_sum - child_set2_sum) + 1)
    child = [child_set1, child_set2, child_fitness, child_set1_sum, child_set2_sum]

    # Randomly mutate if the random float is within the percent chance for mutation
    mutation_happen = random.random()
    if (mutation_happen <= mutation_rate):
        mutate(child, mutation_rate, mutation_happen)

    # Add the child to the new population
    new_population.append(child)
    new_population_fitness += child_fitness

# Tree representation of CKK Algorithm
class node:
    def __init__(self, data, parent, identity):
        self.left = None
        self.right = None
        self.parent = parent
        self.identity = identity
        self.data = data

    def create_tree(self):
        # Create difference branches on the left side
        difference = abs(self.data[0]-self.data[1])
        diff_list = copy.deepcopy(self.data)
        diff_list.remove(self.data[0])
        diff_list.remove(self.data[1])
        diff_list.append(difference)
        diff_list.sort(reverse=True)
        self.left = node(diff_list, self, "diff")
        if len(diff_list) > 1:
            self.left.create_tree()

        # Create sum branches on the right side
        sum = self.data[0]+self.data[1]
        sum_list = copy.deepcopy(self.data)
        sum_list.remove(self.data[0])
        sum_list.remove(self.data[1])
        sum_list.append(sum)
        sum_list.sort(reverse=True)
        self.right = node(sum_list, self, "sum")
        if len(sum_list) > 1:
            self.right.create_tree()

        # Traverse up for the difference branches then for the sum branches
        if len(diff_list) == 1:
            set1 = list()
            set2 = list()
            self.left.traverse_up(self.left, set1, set2, False)
        if len(sum_list) == 1:
            set1 = list()
            set2 = list()
            self.right.traverse_up(self.right, set1, set2, False)

    def traverse_up(self, caller, set1, set2, keep):
        # Traverse back up with the leaf node
        if ((not self.left) and (not self.right)):
            set1.append(self.data[0])
            # Check if leaf node has a better diff than the best diff
            global ckk_best_diff
            if abs(check_best_diff-self.data[0]) < ckk_best_diff:
                ckk_best_diff = abs(check_best_diff-self.data[0])
                keep = True
        else:
            # Traverse back up with a sum or diff branch calling this
            if (caller.identity == "sum"):
                if ((self.data[0] + self.data[1]) in set1):
                    set1.remove(self.data[0] + self.data[1])
                    set1.append(self.data[0])
                    set1.append(self.data[1])
                elif ((self.data[0] + self.data[1]) in set2):
                    set2.remove(self.data[0] + self.data[1])
                    set2.append(self.data[0])
                    set2.append(self.data[1])
            elif (caller.identity == "diff"):
                if (abs(self.data[0] - self.data[1]) in set1):
                    set1.remove(abs(self.data[0] - self.data[1]))
                    set1.append(max(self.data[0], self.data[1]))
                    set2.append(min(self.data[0], self.data[1]))
                elif (abs(self.data[0] - self.data[1]) in set2):
                    set2.remove(abs(self.data[0] - self.data[1]))
                    set1.append(min(self.data[0], self.data[1]))
                    set2.append(max(self.data[0], self.data[1]))

        # If not the root node, then keep traversing up
        # print(self.data, caller.data, set1, set2, keep)
        if self.parent:
            self.parent.traverse_up(self, set1, set2, keep)
        else:
            # If the root node, and the solution is better, set the solutions
            # to the solutions created
            if keep == True:
                global ckk_set1_soln
                global ckk_set2_soln
                ckk_set1_soln = set1
                ckk_set2_soln = set2

    # Print the Tree if needed
    def print_tree(self):
        flag = 0
        if self.left:
            self.left.print_tree()
        print(self.data)
        if self.right:
            self.right.print_tree()

def plot_solution_bar_graph(solution):
    # Basic plotting of the graph
    labels = ['Set 1', 'Set 2']
    sums = [solution[3], solution[4]]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    plot1 = ax.bar(x, sums, width, label='Solution')

    # Set the y label, title, x tick marks, and labels for the ticks
    ax.set_ylabel('Sum')
    ax.set_title('Sum of Set 1 vs. Sum of Set 2')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Label the top of the sets with their sum
    ax.bar_label(plot1, padding=0)

    plt.show()
    plt.clf()

def plot_best_ga_vs_woc(ga_set1_sum, ga_set2_sum, woc_set1_sum, woc_set2_sum):
    # Basic plotting of the graph
    labels = ['Big Set', 'Little Set']
    ga_sums = [ga_set1_sum, ga_set2_sum]
    woc_sums = [woc_set1_sum, woc_set2_sum]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    plot1 = ax.bar(x-width/2, ga_sums, width, label='GA')
    plot2 = ax.bar(x+width/2, woc_sums, width, label='WOC')

    # Set the y label, title, x tick marks, and labels for the ticks
    ax.set_ylabel('Sum')
    ax.set_title('Sum of Sets in Best GA and WOC')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Label the top of the sets with their sum
    ax.bar_label(plot1, padding=0)
    ax.bar_label(plot2, padding=0)

    fig.tight_layout()

    # plt.show()

    save_path = in_file_name.removesuffix(".txt")
    save_path += "seed" + seed_value
    plt.savefig(save_path + ".png")

    plt.clf()

if __name__ == "__main__":
    # Seed random and init variables
    in_file_name = str(sys.argv[1])
    out_file_name = str(sys.argv[2])
    seed_value = str(sys.argv[3])
    random.seed(seed_value)
    number_set = list()
    best_ga_diff = math.inf
    best_ga_set1 = list()
    best_ga_set2 = list()
    best_ga_time = 0
    ga_total_timer = 0
    woc_timer = 0
    text_print_string = ""
    woc_run_flag = False

    # Read the input data file (can be created using CreateData.py)
    read_data(in_file_name)

    woc_timer_start = perf_counter()
    # Create an agreement matrix for use in WOC
    agreement_matrix = np.empty((len(number_set), 3))
    agreement_matrix.fill(0)

    # Figure out if there are duplicates in the number set given for WOC
    number_set_check = set(number_set)
    number_set_check = sorted(number_set_check)
    i = 0
    for number in number_set_check:
        # print(number, ":" ,number_set.count(number))
        for x in range(number_set.count(number)):
            agreement_matrix[i, 0] = number
            i += 1
    woc_timer_finish = perf_counter()
    woc_timer += woc_timer_finish-woc_timer_start

    # Run GA a number of times
    for ga_run in range(0, 25):
        ga_timer = 0
        ga_timer_start = perf_counter()
        # Init GA variables
        population_limit = 10
        number_elites = int(population_limit * .10)
        mutation_rate = float(5/100)
        population = list()
        generation = 0
        improved_difference = math.inf
        gens_without_improvement = 0

        # Initialize the population for GA
        population, population_fitness = init_population(population_limit)
        sort_population(population)
        # print("Gen", generation, "Best Diff:", abs(population[0][3] - population[0][4]))
        generation += 1
        if abs(population[0][3] - population[0][4]) < improved_difference:
            improved_difference = abs(population[0][3] - population[0][4])
            gens_without_improvement = 0
        else:
            gens_without_improvement += 1

        # Run GA for a certain number of generations where there has been no improvement
        while gens_without_improvement < 100:
            # Get elites, create a reproduction pool based on fitness, and reproduce to fill the new population
            # Then make the current population equal to the one just created
            new_population = list()
            new_population_fitness = 0
            get_elites(population, new_population, number_elites, new_population_fitness)
            reproduction_pool = get_reproduction_pool(population, population_fitness)
            while len(new_population) < population_limit:
                reproduce(population, new_population, reproduction_pool, mutation_rate, new_population_fitness)
            population = new_population
            sort_population(population)
            # print("Gen", generation, "Best Diff:", abs(population[0][3] - population[0][4]))
            generation += 1
            if abs(population[0][3] - population[0][4]) < improved_difference:
                improved_difference = abs(population[0][3] - population[0][4])
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1
        ga_timer_finish = perf_counter()
        ga_timer = ga_timer_finish-ga_timer_start
        ga_total_timer += ga_timer

        # Make set 1 the set with the greatest sum and set 2 with the least sum
        if population[0][3] < population[0][4]:
            temp = population[0][0]
            population[0][0] = population[0][1]
            population[0][1] = temp

        # Print out the best solution of the last generation
        print("Final Solution for run: ", ga_run, "\n", population[0], "Diff:", abs(population[0][3] - population[0][4]))

        if (abs(population[0][3] - population[0][4]) < best_ga_diff):
            best_ga_diff = abs(population[0][3] - population[0][4])
            best_ga_set1 = population[0][0]
            best_ga_set2 = population[0][1]
            best_ga_time = ga_timer

        # # Plot a bar graph of the solution
        # plot_solution_bar_graph(population[0])

        woc_timer_start = perf_counter()
        # Fill out the WOC agreement matrix based on which set each instance of a number is found
        for member in population[0:1]:
            i = 0
            for number in number_set_check:
                for j in range(member[0].count(number)):
                    agreement_matrix[i, 1] += 1/(abs(member[3]-member[4]) +1)
                    i += 1
                for j in range(member[1].count(number)):
                    agreement_matrix[i, 2] += 1/(abs(member[3]-member[4]) +1)
                    i += 1
        # print(agreement_matrix)
        woc_timer_finish = perf_counter()
        woc_timer += woc_timer_finish-woc_timer_start

    woc_timer_start = perf_counter()
    woc_set1 = list()
    woc_set2 = list()
    # If the difference of the best set is <=1 (optimal) don't run WOC]
    # Instead, just make WOC solution equal to the best_ga_solution
    if abs(sum(best_ga_set1) - sum(best_ga_set2)) <= 1:
        print("Best GA is optimal")
        woc_set1 = best_ga_set1
        woc_set2 = best_ga_set2
    else:
        woc_run_flag = True
        # Create a WOC solution based on what the agreement is for which set a number should be in
        # If there is less than a certain percentage of members that agree on which set the item goes in
        # Then that item goes in CKK
        ckk_num_list = list()
        number_set.sort()
        for i, number in enumerate(number_set):
            if (agreement_matrix[i,1] > (.50 * (agreement_matrix[i,1] + agreement_matrix[i,2]))):
                woc_set1.append(number)
            elif (agreement_matrix[i,2] > (.50 * (agreement_matrix[i,1] + agreement_matrix[i,2]))):
                woc_set2.append(number)
            else:
                ckk_num_list.append(number)

        # If there are elements for CKK, run CKK
        if len(ckk_num_list) > 0:
            check_best_diff = abs(sum(woc_set1) - sum(woc_set2))
            ckk_best_diff = math.inf
            ckk_set1_soln = list()
            ckk_set2_soln = list()
            # print(check_best_diff)
            root = node(ckk_num_list, None, None)
            root.create_tree()
            # print(ckk_best_diff, ckk_set1_soln, ckk_set2_soln)

            # Find the WOC set with the bigger sum
            if sum(woc_set1) > sum(woc_set2):
                bigger_sum_woc_set = woc_set1
                smaller_sum_woc_set = woc_set2
            else:
                bigger_sum_woc_set = woc_set2
                smaller_sum_woc_set = woc_set1
            # print(bigger_sum_woc_set, smaller_sum_woc_set)

            # Find the CKK set with the bigger sum
            if sum(ckk_set1_soln) > sum(ckk_set2_soln):
                bigger_sum_ckk_set = ckk_set1_soln
                smaller_sum_ckk_set = ckk_set2_soln
            else:
                bigger_sum_ckk_set = ckk_set2_soln
                smaller_sum_ckk_set = ckk_set1_soln
            # print(bigger_sum_ckk_set, smaller_sum_ckk_set)

            # Put the bigger sum set elements of CKK into the smaller sum set for WOC
            # and the smaller sum set elements of CKK into the bigger sum set for WOC
            for number in bigger_sum_ckk_set:
                smaller_sum_woc_set.append(number)
            for number in smaller_sum_ckk_set:
                bigger_sum_woc_set.append(number)
            # print(abs(sum(bigger_sum_woc_set) - sum(smaller_sum_woc_set)), bigger_sum_woc_set, smaller_sum_woc_set)
    woc_timer_finish = perf_counter()
    woc_timer += woc_timer_finish-woc_timer_start

    # Sort the WOC and GA sets for printing
    best_ga_set1.sort(reverse=True)
    best_ga_set2.sort(reverse=True)
    woc_set1.sort(reverse=True)
    woc_set2.sort(reverse=True)

    # Information about the run to print to a file and terminal
    text_print_string += "File: " + str(in_file_name) + "\n"
    text_print_string += "Random Seed Value: " + str(seed_value) + "\n"
    text_print_string += "WOC Agreement Matrix Run: " + str(woc_run_flag) + "\n"
    text_print_string += "Best GA Solution: " + str(abs(sum(best_ga_set1) - sum(best_ga_set2))) + "\nSet1: " + str(best_ga_set1) + "\nSet2: " + str(best_ga_set2) + "\n"
    text_print_string += "WOC Solution: " + str(abs(sum(woc_set1) - sum(woc_set2))) + "\nSet1: " + str(woc_set1) + "\nSet2: " + str(woc_set2) + "\n"
    text_print_string += "Timer info: \nBest GA Time: " + str(best_ga_time) + "\nTotal GA Time: " + str(ga_total_timer) + "\nWOC Total Time: " + str(woc_timer) + "\n\n"
    print(text_print_string)
    out_file = open(out_file_name, 'a')
    out_file.write(text_print_string)
    out_file.close()

    # Plot the best GA and the WOC solutions
    max_ga_set = max(sum(best_ga_set1), sum(best_ga_set2))
    min_ga_set = min(sum(best_ga_set1), sum(best_ga_set2))
    max_woc_set = max(sum(woc_set1), sum(woc_set2))
    min_woc_set = min(sum(woc_set1), sum(woc_set2))
    plot_best_ga_vs_woc(max_ga_set, min_ga_set, max_woc_set, min_woc_set)
