# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import geatpy as ea  # import geatpy
from getdata import get_data
import copy
from math import sqrt
from Classical_TSP import *
file='TSPTW_dataset.txt'
places = get_data(file)
places_2=copy.deepcopy(places)

#add=np.linspace(100,100,100)
#add= np.zeros((100,2), dtype = [['x', 'i4'], ['y', 'i4']])
for i in range(len(places_2)):
        result=100+float(places_2[i][0])
        places_2[i][0]=result
        #print(places_2[i])
places_new=places+places_2
# for i in range(len(places_new)):
#     for j in range(len(places_new[i])):
#         float(places_new[i][j])   无用的部分 这种写法错误的

#print(type(places_new[0][0]))



#places_add=np.add(places,add)


print(places_new)
print(len(places_new))


def get_distance(position1, position2):
    '''
        获取相邻两个城市的距离
        position1:第一个地区的经纬度，为列表形式

        position2:第二个地区的经纬度，为列表形式

        传入两个地区的地理坐标，返回二者的距离
    '''
    lng1, lat1, lng2, lat2 = (position1[0], position1[1], position2[0], position2[1])
    # print(lng1,lng2,lat1,lat2)
    # lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    point = [lng1, lat1, lng2, lat2]
    lng1, lat1, lng2, lat2 = map(float, point)
    #print(lng1, lng2, lat1, lat2)
    # lng1, lat1, lng2, lat2 = map(float,lng1, lat1, lng2, lat2)
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = dlon ** 2 + dlat ** 2
    distance = round(sqrt(a))
    return distance
class MyProblem2(ea.Problem):  # 继承Problem父类

    def __init__(self):
        # 目标函数计算中用到的一些数据
        #self.datas = np.loadtxt('data.csv', delimiter=',')
        self.datas = np.array(places_new,dtype=float)# 读取数据
        self.k = 4  # 分类数目
        # 问题类设置
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = self.datas.shape[1] * self.k  # 初始化Dim
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = list(np.min(self.datas, 0)) * self.k  # 决策变量下界
        ub = list(np.max(self.datas, 0)) * self.k  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.places = np.array(places_new)
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
        centers = pop.Phen.reshape(int(pop.sizes * self.k),
                                   int(pop.Phen.shape[1] / self.k))  # 得到聚类中心
        dis = ea.cdist(centers, self.datas, 'euclidean')  # 计算距离
        dis_split = dis.reshape(
            pop.sizes, self.k,
            self.datas.shape[0])  # 分割距离矩阵，把各个聚类中心到各个点之间的距离的数据分开
        labels = np.argmin(dis_split, 1)[0]  # 得到聚类标签值
        uni_labels = np.unique(labels)
        for i in range(len(uni_labels)):
            centers[uni_labels[i], :] = np.mean(
                self.datas[np.where(labels == uni_labels[i])[0], :], 0)
        # 直接修改染色体为已知的更优值，加快收敛
        pop.Chrom = centers.reshape(pop.sizes, self.k * centers.shape[1])
        pop.Phen = pop.decoding()  # 染色体解码（要同步修改Phen，否则后面会导致数据不一致）
        dis = ea.cdist(centers, self.datas, 'euclidean')
        dis_split = dis.reshape(pop.sizes, self.k, self.datas.shape[0])
        pop.ObjV = np.sum(np.min(dis_split, 1), 1, keepdims=True)  # 计算个体的目标函数值

        """得到了centers之后开始分开对不同组进行求路线"""

    def draw(self, centers):  # 绘制聚类效果图
        dis = ea.cdist(centers, self.datas, 'euclidean')
        dis_split = dis.reshape(1, self.k, self.datas.shape[0])
        labels = np.argmin(dis_split, 1)[0]
        colors = ['r', 'g', 'b', 'y']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.k):
            idx = np.where(labels == i)[0]  # 找到同一类的点的下标
            datas = self.datas[idx, :]
            ax.scatter(datas[:, 0], datas[:, 1], c=colors[i])

"""求解路径的类"""
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
        self.places = np.array(places_new)

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen.copy()  # 得到决策变量矩阵
        # 添加最后回到出发地
        X = np.hstack([x, x[:,[0]]]).astype(int)
        ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(pop.sizes):
            journey = self.places[X[i], :]  # 按既定顺序到达的地点坐标
            distance = 0.0
            for j in range(len(journey)-1):
                distance+=get_distance(journey[j],journey[j+1])
            # distance = np.sum(np.sqrt(np.sum(np.diff(journey.T) ** 2, 0)))  # 计算总路程
            # print("dis",distance)
            ObjV.append(distance)
        pop.ObjV = np.array([ObjV]).T

"""利用进化算法进行仿k-means聚类（EA-KMeans算法）.

采用与k-means类似的聚类方法，采用展开的聚类中心点坐标作为染色体的编码，基本流程大致如下：
1) 初始化种群染色体。
2) 迭代进化（循环第3步至第6步），直到满足终止条件。
3) 重组变异，然后根据得到的新染色体计算出对应的聚类中心点。
4) 计算各数据点到聚类中心点的欧式距离。
5) 把与各中心点关联的数据点的坐标平均值作为新的中心点，并以此更新种群的染色体。
6) 把各中心点到与其关联的数据点之间的距离之和作为待优化的目标函数值。
注意：导入的数据是以列为特征的，即每一列代表一个特征（如第一列代表x，第二列代表y......）。
"""


if __name__ == '__main__':
    # 实例化问题对象
    problem2 = MyProblem2()
    # 构建算法
    algorithm = ea.soea_DE_rand_1_bin_templet(
        problem2,
        ea.Population(Encoding='RI', NIND=1000),
        MAXGEN=500,  # 最大进化代数。
        logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
        trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
        maxTrappedCount=10)  # 进化停滞计数器最大上限值。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True)
    # 检验结果
    if res['success']:
        print('最优的聚类中心为：')
        Vars = res['Vars'][0, :]
        centers = Vars.reshape(problem2.k,
                               int(len(Vars) / problem2.k))  # 得到最优的聚类中心
        print(centers)
        """=========================分组求路径啦===================================="""
    if problem2.k==4 :
        route_point=[places_new[0:50],places_new[50:100],places_new[100:150],places_new[150:200]]
    elif problem2.k==2 :
        route_point = [places_new[0:100], places_new[100:200]]
    else:
        pass

    #print('disizu:',route_point[3])
    #print(places_new)
    #print(centers)
    journey_final=[]
    total_dis = 0
    for k in range(len(centers)):

        places=route_point[k]
        centers_point=centers[k]
        print(len(places))
        problem = MyProblem()  # 生成问题对象
        """==================================种群设置=============================="""
        Encoding = 'P'  # 编码方式，采用排列编码
        NIND = 1000  # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
        population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        """================================算法参数设置============================="""
        myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
        myAlgorithm.MAXGEN = 500  # 最大进  代数
        myAlgorithm.recOper.XOVR = 0.5  # 重组概率
        myAlgorithm.mutOper.Pm = 0.2  # 变异概率
        trappedValue = 1e-6  # 单目标优化陷入停滞的判断阈值。
        maxTrappedCount = 10
        myAlgorithm.logTras = 0  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.verbose = True  # 设置是否打印输出日志信息
        myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        """===========================调用算法模板进行种群进化========================"""
        [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
        BestIndi.save()  # 把最优个体的信息保存到文件中
        """==================================输出结果=============================="""
        print('第%s个区域结果'%(k+1))
        print('第%s个区域的评价次数：%s'%(k+1,myAlgorithm.evalsNum))
        print('时间已过 %s 秒' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            print('第%s个区域的最短路程为：%s' % (k+1, BestIndi.ObjV[0][0]))
            print('第%s个区域的最佳路线为：' %(k+1))
            best_journey = np.hstack([0, BestIndi.Phen[0, :]])
            total_dis = total_dis + BestIndi.ObjV[0][0]

            best_route=[]
            for i in range(len(best_journey)):
                best_route_s = best_journey[i] + k * 50
                best_route.append(best_route_s)
                print(int(best_route_s), end=' ')

            print()
            journey_part = best_route[0:-1]
            journey_final=journey_final+best_route
            #total_dis.append(BestIndi.ObjV[0][0])
            #print(journey_part)

        else:
            print('no find the results can be used')

        """=================================检验结果==============================="""
        problem2.draw(centers)
        print(len(journey_part))

    print('此算法最终的路径为：',journey_final)
    print('路径距离是',total_dis)


