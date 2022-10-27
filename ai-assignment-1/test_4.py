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
np.array(places)
profit0 = get_profit()
profit0_add=[]
for i in range(100):
    a=0
    profit0_add.append(a)

profit1=np.array(list(zip(profit0,profit0_add)))
#profit1=np.array(list(profit0))
#print(profit1)

#print(len(profit1))
for i in range(len(places)):
    for j in range(len(places[i])):
        a = places[i][j]
        #int(float(a))
        places[i][j]=int(float(a))
#print(places)
#合并两个矩阵为一个矩阵使用
input = np.hstack((places,profit1))
#input=places+profit0
#print(input)

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

class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = len(places)-1  # 初始化Dim（决策变量维数）
        maxormins = [1]*M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [Dim] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.data=np.array(places)
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)


    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen.copy()  # 得到决策变量矩阵
        ObjV1 = []
        ObjV2 = []
        profit0_0=profit0.copy()
        profit1=profit0+[profit0_0[0]]
        # 添加最后回到出发地
        X = np.hstack([x, x[:, [0]]]).astype(int)
        #ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(pop.sizes):
            journey = self.data[X[i], :]  # 按既定顺序到达的地点坐标
            distance = 0.0
            profit = 0.0
            for j in range(len(journey) - 1):
                distance += get_distance(journey[j], journey[j + 1])
                profit+= random.randint(1,50)
                #profit += profit0[j]+random.randint(1,25) #随机处理每个客户的销售额
            profit = 1/profit
            ObjV1.append(distance)
            ObjV2.append(profit)
        ObjV1 = np.array([ObjV1]).T
        ObjV2 = np.array([ObjV2]).T
        ObjV = np.hstack([ObjV1, ObjV2])
        pop.ObjV = ObjV  #得到目标矩阵
        print('ceshishuju',ObjV,len(ObjV))


if __name__ == '__main__':
    #  实例化问题对象
    problem = MyProblem()
    """==================================种群设置=============================="""
    # 定义outFunc()函数
    def outFunc(alg, pop): # alg 和 pop为outFunc的固定输入参数，分别为算法对象和每次迭代的种群对象。

        print('第 %d 代' % alg.currentGen)

    # 构建算法
    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='RI', NIND=1000),
        MAXGEN=500,  # 最大进化代数
        logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
        outFunc=outFunc)
    # 求解

    res = ea.optimize(algorithm,
                      verbose=False,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=False)
    total_price=[]
    for i in res['ObjV']:
        rs=[i[0],1/i[1]]
        total_price.append(rs)
    print('dis,price',total_price)
    #print(1/(res['ObjV'][1]),type(res['ObjV']))
    print(res)