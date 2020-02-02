import matplotlib.pyplot as plt
import numpy as np

#设置中文和'-'号显示问题
from pylab import mpl

mpl.rcParams['font.sans-serif']=['FangSong']
mpl.rcParams['axes.unicode_minus']=False
'''
#获得figure对象

fig=plt.figure(figsize=(8,6))

#在Figure对象上创建axes对象
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)

#在当前axes上绘制曲线图(ax3)
plt.plot(np.random.randn(50).cumsum(),'k--')
#在ax1上绘制柱状图
ax1.hist(np.random.randn(300),bins=20,color='k',alpha=0.3)
#在ax2上绘制散点图
ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30))

fig,axes=plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500),bins=10,color='k',alpha=0.5)
    plt.subplots_adjust(wspace=0,hspace=0)

x=np.linspace(0,10,1000)
y=np.sin(x)
z=np.cos(x**2)
plt.figure(figsize=(8,4),dpi=100)

#label:给所绘制的曲线一个名字,此名字在图示(legend)中显示
#只要在字符串前后添加"$"符号,matplotlib就会使用其内嵌的latex引擎绘制的数学公式
#color指定曲线颜色,linewidth曲线宽度,"b--指定曲线的颜色和线型"

plt.plot(x,y,label='$sin(x)$',color="red",linewidth=2)
plt.plot(x,z,"b--",label="$cos(x^2)$")

#设置x轴标题
plt.xlabel("Time(s)")
#设置y轴标题
plt.ylabel("Volt")
#设置图标标题
plt.title("小马 First Example")
#设置x轴范围
plt.ylim(-1.2,1.2)
#显示图示说明
plt.legend()
#显示虚线框
plt.grid(True)
#保存图片
plt.savefig("test.png",dpi=120)
#展示图表
plt.show()
'''


#散点图

'''
plt.axis([0,5,0,20])
plt.title('My First Chart',fontsize=20,fontname='Times New Roman')
plt.xlabel('Counting',color='gray')
plt.ylabel('Square values',color='gray')
plt.text(1,1.5,'First')
plt.text(2,4.5,'Second')
plt.text(1,11.5,r'$y=x^2$',fontsize=20,bbox={'facecolor':'yellow','alpha':0.2})
plt.grid(True)
plt.plot([1,2,3,4],[1,4,9,6],'ro')
plt.plot([1,2,3,4],[0.8,3.5,8,15],'g^')
plt.legend(['First series','Second series','Third series'],loc=2)
help(plt.plot)
plt.show()

'''

plt.subplot(221)
plt.subplot(222)
plt.subplot(212)



plt.show()