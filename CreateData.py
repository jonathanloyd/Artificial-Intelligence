#Authors: Jonathan Loyd, Caty Battjes, Nicole Hartman
#Description: Python3 Create Data for Number Partitioning Problem
#CSE545 Project 6

import sys, random

def write_file(outFilename, number_of_numbers, number_range_lower, number_range_upper):
    outFile = open(outFilename, 'w')
    for number in range(number_of_numbers):
        write_string = str(random.randint(number_range_lower, number_range_upper)) + "\n"
        outFile.write(write_string)
    outFile.close()

if __name__ == "__main__":
    outFilename = str(sys.argv[1])
    number_of_numbers = int(sys.argv[2])
    number_range_lower = int(sys.argv[3])
    number_range_upper = int(sys.argv[4])

    write_file(outFilename, number_of_numbers, number_range_lower, number_range_upper)
