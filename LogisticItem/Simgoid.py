import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-10,10,100)

y=1/(1+np.exp(x*(-1)))
plt.plot(x,y)
plt.show()