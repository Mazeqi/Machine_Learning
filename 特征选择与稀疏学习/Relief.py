# An highlighted block
# _*_ coding:utf8 _*_

"""
# 说明：特征选择方法一：过滤式特征选择（ReliefF算法）
# 思想：先用特征选择过程对初始特征进行"过滤"，然后再用过滤后的特征训练模型
# 时间：2019-1-16
# 问题：
"""

import pandas as pd
import numpy as np
import numpy.linalg as la
import random


# 异常类
class ReliefError:
    pass


class Relief:
    def __init__(self, data_df, sample_rate, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例
        :param t: 统计量分量阈值
        :param k: k近邻的个数
        """
        self.__data = data_df
        self.__feature = data_df.columns
        self.__sample_num = int(round(len(data_df) * sample_rate))
        self.__t = t
        self.__k = k

    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def get_data(self):
        new_data = pd.DataFrame()
        for one in self.__feature[:-1]:
            col = self.__data[one]
            if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():
                new_data[one] = self.__data[one]
                # print '%s 是数值型' % one
            else:
                # print '%s 是离散型' % one
                keys = list(set(list(col)))
                values = list(range(len(keys)))
                new = dict(zip(keys, values))
                new_data[one] = self.__data[one].map(new)
        new_data[self.__feature[-1]] = self.__data[self.__feature[-1]]
        return new_data

    # 返回一个样本的k个猜中近邻和其他类的k个猜错近邻
    def get_neighbors(self, row):
        df = self.get_data()
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        f = lambda x: eulidSim(np.mat(x), np.mat(aim))
        right_sim = right_df.apply(f, axis=1)
        right_sim_two = right_sim.drop(right_sim.idxmin())
        right = dict()
        right[row_type] = list(right_sim_two.sort_values().index[0:self.__k])
        # print list(right_sim_two.sort_values().index[0:self.__k])
        types = list(set(df[df.columns[-1]]) - set([row_type]))
        wrong = dict()
        for one in types:
            wrong_df = df[df[df.columns[-1]] == one].drop(columns=[df.columns[-1]])
            wrong_sim = wrong_df.apply(f, axis=1)
            wrong[one] = list(wrong_sim.sort_values().index[0:self.__k])
        print (right, wrong)
        return right, wrong

    # 计算特征权重
    def get_weight(self, feature, index, NearHit, NearMiss):
        # data = self.__data.drop(self.__feature[-1], axis=1)
        data = self.__data
        row = data.iloc[index]
        right = 0
        for one in NearHit.values()[0]:
            nearhit = data.iloc[one]
            if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                max_feature = data[feature].max()
                min_feature = data[feature].min()
                right_one = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
            else:
                right_one = 0 if row[feature] == nearhit[feature] else 1
            right += right_one
        right_w = round(right / self.__k, 2)

        wrong_w = 0
        # 样本row所在的种类占样本集的比例
        p_row = round(float(list(data[data.columns[-1]]).count(row[data.columns[-1]])) / len(data), 2)
        for one in NearMiss.keys():
            # 种类one在样本集中所占的比例
            p_one = round(float(list(data[data.columns[-1]]).count(one)) / len(data), 2)
            wrong_one = 0
            for i in NearMiss[one]:
                nearmiss = data.iloc[i]
                if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                    max_feature = data[feature].max()
                    min_feature = data[feature].min()
                    wrong_one_one = pow(round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2), 2)
                else:
                    wrong_one_one = 0 if row[feature] == nearmiss[feature] else 1
                wrong_one += wrong_one_one

            wrong = round(p_one / (1 - p_row) * wrong_one / self.__k, 2)
            wrong_w += wrong
        w = wrong_w - right_w
        return w

    # 过滤式特征选择
    def reliefF(self):
        sample = self.get_data()
        # print sample
        m, n = np.shape(self.__data)  # m为行数，n为列数
        score = []
        sample_index = random.sample(range(0, m), self.__sample_num)
        print ('采样样本索引为 %s ' % sample_index)
        num = 1
        for i in sample_index:    # 采样次数
            one_score = dict()
            row = sample.iloc[i]
            NearHit, NearMiss = self.get_neighbors(row)
            print('第 %s 次采样，样本index为 %s，其NearHit k近邻行索引为 %s ，NearMiss k近邻行索引为 %s' % (num, i, NearHit, NearMiss))
            for f in self.__feature[0:-1]:
                w = self.get_weight(f, i, NearHit, NearMiss)
                one_score[f] = w
                print ('特征 %s 的权重为 %s.' % (f, w))
            score.append(one_score)
            num += 1
        f_w = pd.DataFrame(score)
        print ('采样各样本特征权重如下：')
        print (f_w)
        print ('平均特征权重如下：')
        print (f_w.mean())
        return f_w.mean()

    # 返回最终选取的特征
    def get_final(self):
        f_w = pd.DataFrame(self.reliefF(), columns=['weight'])
        final_feature_t = f_w[f_w['weight'] > self.__t]
        print (final_feature_t)
        # final_feature_k = f_w.sort_values('weight').head(self.__k)
        # print final_feature_k
        return final_feature_t


# 几种距离求解
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)


def cosSim(vecA, vecB):
    """
    :param vecA: 行向量
    :param vecB: 行向量
    :return: 返回余弦相似度（范围在0-1之间）
    """
    num = float(vecA * vecB.T)
    denom = la.norm(vecA) * la.norm(vecB)
    cosSim = 0.5 + 0.5 * (num / denom)
    return cosSim


def pearsSim(vecA, vecB):
    if len(vecA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB, rowvar=0)[0][1]


if __name__ == '__main__':
    data = pd.read_csv('water3.0.csv')[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']]
    print (data)
    f = Relief(data, 1, 0.2, 2)
    # df = f.get_data()
    # print type(df.iloc[0])
    # f.get_neighbors(df.iloc[0])
    # f.get_weight('色泽', 6, 7, 8)
    f.reliefF()
    # f.get_final()

