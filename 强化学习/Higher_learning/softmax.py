#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def R(k, P, V):
    if random.random() < P[k]:
        return V[k]
    else:
        return 0


def eplison_bandit(K, P, V, R, T, tau=0.1):
    r = 0
    Q = np.zeros(K)
    count = np.zeros(K)
    for t in range(T):
        p = softmax(Q / tau)
        rand = random.random()
        total = 0.0
        for i in range(K):
            total += p[i]
            if total >= rand:
                k = i
                break
        v = R(k, P, V)
        r += v
        Q[k] = (Q[k]*count[k]+v)/(count[k]+1)
        # Q[k] += (v - Q[k]) / (count[k] + 1)
        count[k] += 1
    return r


def main():
    K = 5
    P = np.array([0.1, 0.9, 0.3, 0.2, 0.7])
    V = np.array([5, 3, 1, 7, 4])
    T = 1000000
    tau = 0.1
    print (eplison_bandit(K, P, V, R, T, tau))

if __name__ == '__main__':
    main()