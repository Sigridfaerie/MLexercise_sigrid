#assignment1
#no.1
#by Sigrid Oct.2022


import geatpy as ea
import numpy as np
from math import sin, asin, cos, radians, fabs, sqrt
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
from getdata import get_data

file='TSPTW_dataset.txt'
places = get_data(file)
#places = map(eval(places))
#print(places)
# TSP坐标点
#places =[[117.000923, 36.675807], [115.48333, 38.03333],[113.5, 22.2]]


# print(len(places))
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

# 定义问题
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'Classical TSP(first problem)'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = len(places) - 1 # 初始化  im（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1] * Dim  # 决策变量下界
        ub = [Dim] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 新增一个属性存储旅行地坐标
        self.places = np.array(places)

    def aimFunc(self, pop):  # 目标函数
        #print(pop)
        x = pop.Phen.copy()  # 得到决策变量矩阵
        # 添加最后回到出发地
        X = np.hstack([x, x[:, [0]]]).astype(int)
        ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(pop.sizes):
            journey = self.places[X[i], :]  # 按既定顺序到达的地点坐标
            distance = 0.0
            for j in range(len(journey)-1):
                distance+=get_distance(journey[j],journey[j+1]) #总路程
            # distance = np.sum(np.sqrt(np.sum(np.diff(journey.T) ** 2, 0)))  # 计算总路程
            # print("dis",distance)
            ObjV.append(distance)
        print('测试数据',np.array([ObjV]).T)
        pop.ObjV = np.array([ObjV]).T


# if __name__ == '__main__':
#     pass
#     print(places[0],places[1])
#     print(get_distance(places[0],places[1]))

"""调用模板求解"""
if __name__ == '__main__':
    """================================实例化问题对象============================"""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'P'  # 编码方式，采用排列编码
    NIND = 10 # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 50 # 最大进  代数
    myAlgorithm.recOper.XOVR = 0.9  # 重组概率
    myAlgorithm.mutOper.Pm = 0.5  # 变异概率
    trappedValue = 1e-6  # 单目标优化陷入停滞的判断阈值。
    maxTrappedCount = 10
    myAlgorithm.logTras = 100  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """==================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最短路程为：%s' % BestIndi.ObjV[0][0])
        print('最佳路线为：')
        best_journey = np.hstack([0, BestIndi.Phen[0, :], 0])
        for i in range(len(best_journey)):
            print(int(best_journey[i]), end=' ')
        print()
        # 绘图
        plt.figure(figsize=(12, 10))
        plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], c='black')
        plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], 'o',
                 c='black')
        for i in range(len(best_journey)):
            # plt.text(problem.places[int(best_journey[i]), 0], problem.places[int(best_journey[i]), 1],
            # chr(int(best_journey[i]) + 65), fontsize=20)
            plt.text(problem.places[int(best_journey[i]), 0], problem.places[int(best_journey[i]), 1],
                     int(best_journey[i]), fontsize=10)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('roadmap01.svg', dpi=600, bbox_inches='tight')
    else:
        print('no find the results can be used')

