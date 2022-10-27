#deal the data
#by Sigrid
import math
import numpy as np
from collections import Counter
import warnings
file='TSPTW_dataset.txt'
def get_data(filename):
        dataset=[]
        f=open(filename,'r', encoding='UTF-8')
        lines=f.readlines()
        #get line to
        for i in range(1,len(lines)):
            line = lines[i].strip().split()
            dataset.append(line)
        f.close()
        spots = []
        for j in range(len(dataset)):
            spot = [dataset[j][1],dataset[j][2]]
            spots.append(spot)
        #print(spots)
        return spots


def get_time(filename): #获得txt文件中的readytime 和 duetime
    dataset = []
    f = open(filename, 'r', encoding='UTF-8')
    lines = f.readlines()
    # get line to
    for i in range(1, len(lines)):
        line = lines[i].strip().split()
        dataset.append(line)
    f.close()
    times = []
    for j in range(len(dataset)):
        spot = [dataset[j][4], dataset[j][5]]
        times.append(spot)
    # print(spots)
    return times

# if __name__ == '__main__':
#     get_data(file)
#     print(len(get_data(file)))
print(get_time(file))