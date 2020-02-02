import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

mpl.rcParams['font.sans-serif']=['FangSong']
mpl.rcParams['axes.unicode_minus']=False

x=np.linspace(0.05,10,1000)
y=np.sin(x)

#plt.scatter(x,y,c="y",label="scatter")
plt.plot(x,y,ls="-.",lw=2,c="c",label="plot")
plt.xlabel("x轴")
plt.ylabel("y轴")
#plt.axvspan(xmin=4.0,xmax=6.0,facecolor="y",alpha=1.0)

plt.legend()
plt.annotate(
    "mazeqi",
    xy=(np.pi/2,1.0),
    xytext=((np.pi/2)+1,.8),
    weight="bold",
    color="b",
    arrowprops=dict(arrowstyle="",connectionstyle="arc3",color="b")
)

plt.grid(linestyle=":",color="r")
plt.show()