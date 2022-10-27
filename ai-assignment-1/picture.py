from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图


#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#定义图像和三维格式坐标轴
# fig=plt.figure()
# ax2 = Axes3D(fig)

xs1 = np.random.randint(30, 40, 100)
ys1 = np.random.randint(20, 30, 100)
zs1 = np.random.randint(10, 20, 100)
xs2 = np.random.randint(50, 60, 100)
ys2 = np.random.randint(30, 40, 100)
zs2 = np.random.randint(50, 70, 100)
xs3 = np.random.randint(10, 30, 100)
ys3 = np.random.randint(40, 50, 100)
zs3 = np.random.randint(40, 50, 100)
x=[xs1,xs2,xs3]
y=[ys1,ys2,ys3]
z=[zs1,zs2,zs3]
import numpy as np
# z = np.linspace(0,13,1000)
# x = 5*np.sin(z)
# y = 5*np.cos(z)
# zd = 13*np.random.random(100)
# xd = 5*np.sin(zd)
# yd = 5*np.cos(zd)
# ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
ax1.scatter3D(x,y,z, c='r',cmap='Blues')
#ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()

