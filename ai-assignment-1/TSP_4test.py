"""快崩溃的某人写不下去了嘻嘻"""
import numpy as np

import geatpy as ea
import random
from getdata import get_data
from math import sqrt

file='TSPTW_dataset.txt'
places = get_data(file)
#get the data of profit
def get_profit():
    profit=[]
    for i in range(100):
        p=random.randint(1,50)
        profit.append(p)
    return profit

profit0 = get_profit()
print(profit0)
#print(profit(),len(profit()))
def get_distance(position1 ,position2):
    '''
        获取相邻两个城市的距离
        position1:第一个地区的经纬度，为列表形式

        position2:第二个地区的经纬度，为列表形式

        传入两个地区的地理坐标，返回二者的距离
    '''
    lng1 ,lat1 ,lng2 ,lat2 = (position1[0] ,position1[1] ,position2[0] ,position2[1])
    #print(lng1,lng2,lat1,lat2)
   # lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    point=[lng1, lat1, lng2, lat2]
    lng1, lat1, lng2, lat2=map(float,point)
    #print(lng1,lng2,lat1,lat2)
    #lng1, lat1, lng2, lat2 = map(float,lng1, lat1, lng2, lat2)
    dlon =lng2 -lng1
    dlat =lat2 -lat1
    a=dlon**2+dlat**2
    distance = round(sqrt(a))
    return distance
#distance=get_distance(places)
#print(distance)



class MyProblem(ea.Problem):

    def __init__(self):
        name = 'MyProblem'
        M = 2
        maxormins = [1,1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = len(places) - 1  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        #决策变量上下边界
        lb = [1] * Dim
        ub = [Dim] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 新增属性存储坐标和利润
        self.data = np.array(places)
        self.weight = 0

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen.copy()  # 得到决策变量矩阵
        ObjV1 = []
        ObjV2 = []
        # 添加最后回到出发地
        X = np.hstack([x, x[:, [0]]]).astype(int)
        ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(pop.sizes):
            journey = self.data[X[i], :]  # 按既定顺序到达的地点坐标
            distance = 0.0
            profit = 0.0
            for j in range(len(journey) - 1):
                distance += get_distance(journey[j], journey[j + 1])
                #profit += get_profit(journey[j],journey[j+1])
                profit +=profit0[j]
            #profit = 1/profit
            ObjV1.append(distance)
            ObjV2.append(profit)
        ObjV1 = np.array([ObjV1]).T
        ObjV2 = np.array([ObjV2]).T
        ObjV = np.hstack([ObjV1, ObjV2])
        pop.ObjV = ObjV


if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='BG', NIND=30),

        MAXGEN=300,  # 最大进化代数
        logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=False,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=False)
    print(res)



