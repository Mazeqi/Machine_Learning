import numpy as np
import pandas as pd

def ConstrainedKMeans(X,K,label,test=None,max_iter=100):
    """
    X为数据集,K为聚类的簇的数目,label为标签，大多数为NaN
    """

    LabelIndex = np.where(np.isfinite(label))[0]  # 带有label的下标
    M = {keys:np.setdiff1d(np.where(label==(label[keys]))[0],keys) for keys in LabelIndex}  # 必连关系
    C = {keys:np.where((label!=(label[keys]))&np.isfinite(label))[0] for keys in LabelIndex}  # 不连关系
    Mean = X.iloc[np.random.randint(0,len(X),K),:]  # 初始化均值
    Mean.index = ['Class'+str(k+1) for k in range(K)]
    Class = np.NAN*np.zeros(len(X))  # 各个样本所属的类
    for i in range(max_iter):
        # 样本到中心的距离
        dist = pd.concat([np.sqrt(((X-Mean.iloc[k,:])**2).sum(axis=1)) for k in range(K)],axis=1)
        for n in range(len(dist)):  # 遍历所有样本
            is_merged = False
            while not is_merged:  # 循环,直到找到满足条件的类
                nearest = np.argmin(dist.values[n])  # 属于最近距离的类
                if n in M.keys():
                    lim = M[n]
                    if not all(np.in1d(lim[lim<n],np.where(Class==nearest)[0])):  # 是否满足条件M
                        dist.values[n][nearest]=np.inf   # 不满足条件，则不能属于该类，距离改为无穷
                        continue
                if n in C.keys():
                    lim = C[n]
                    if any(np.in1d(lim[lim<n],np.where(Class==nearest)[0])):  # 是否满足条件C
                        dist.values[n][nearest]=np.inf   # 不满足条件，则不能属于该类，距离改为无穷
                        continue
                Class[n] = nearest
                is_merged = True
        Mean_new = np.array([X[Class == k].mean().values for k in range(K)])  # 更新均值
        if all((Mean.values == Mean_new).ravel()):  # 均值不发生改变
            break
        else:
            Mean.values[:,:]=Mean_new
    return Class, Mean