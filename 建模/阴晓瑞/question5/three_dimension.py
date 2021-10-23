from matplotlib import mlab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#定义坐标轴
fig = plt.figure()

ax = fig.add_subplot(999,projection='3d')  #这种方法也可以画多个子图

#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义图像和三维格式坐标轴
fig=plt.figure()
ax = Axes3D(fig)
def get_x_y_list():
    with open('./55z.txt','r') as f:
        contents = f.readlines()
    x_list = []
    y_list = []
    z_list = []
    for sample in contents:
        sample_data = sample.strip().split()
        x = float(sample_data[0])/5000
        y = float(sample_data[1])/5000
        z = float(sample_data[2])/5000
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    return x_list,y_list,z_list
x_list,y_list,z_list = get_x_y_list()

import numpy as np
z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax.scatter3D(x_list,y_list,z_list, cmap='Blues')  #绘制散点图
# ax.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()
plt.show()
