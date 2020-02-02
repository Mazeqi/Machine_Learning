import numpy as np
import pandas as pd


def ConstrainedSeedKMeans(X,K,label,test=None,max_iter=100):
    """ 
    X为数据集,K为聚类的簇的数目,label为标签
    """
    LabelIndex = np.where(np.isfinite(label))[0]  # 带有label的下标
    Mean = X.iloc[np.random.randint(0,len(X), K), :]  # 初始化均值
    Mean.index = ['Class'+str(k+1) for k in range(K)]

    for Label in np.unique(label[np.isfinite(label)]):
        Mean.values[int(Label)] = X[label == Label].mean()

    Class = label  # 各个样本所属的类

    for i in range(max_iter):
        dist = pd.concat([np.sqrt(((X-Mean.iloc[k,:])**2).sum(axis=1)) for k in range(K)],axis=1)
        for n in range(len(dist)):
            if np.isnan(label[n]):  # 不带标签的样本
                Class[n]=np.argmin(dist.values[n])

        Mean_new = np.array([X[Class == k].mean().values for k in range(K)])  # 更新均值

        if all((Mean.values == Mean_new).ravel()):
            break
        else:
            Mean.values[:,:] = Mean_new

    return Class, Mean