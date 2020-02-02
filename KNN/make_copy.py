from numpy import *
import operator


#反斜杠是换行符,表示可以换行接着上一行的代码
'''
ndarry=array\
(
    [
        [
            [1,2,3,4,5],
            [5,4,3,5,2],
            [7, 8, 9, 6,1]
        ],
        [
            [1,3,2,6,7],
            [7,8,9,6,1],
            [5, 4, 3, 5, 2]
        ]
    ],dtype=int
)
print(sorted(ndarry[0][1],reverse=True))

print(ndarry[0,1])

#数组的维度数
print(ndarry.ndim)

#数组的类型
print(ndarry.dtype)

#数组的形状
print(ndarry.shape)

#数组中数字的数目
print(ndarry.size)
'''



'''

##生成0 生成1 生成空数组
print(zeros((3,4)))
print(ones((3,4)))
print(empty((2,3,4)))


print(arange(20))

#生成从一到二十,且步长为2的数组
print(arange(1,20,2))

#reshape()函数使数组重新构成,但是数组元素不变
a=arange(0,12)
print(a.reshape(3,4))

#输出0到10之间的五个数,且最终值为时,即步长为10/4,这里是等差数列
print(linspace(0,10,5))
print(linspace(0,20,5))
print(help(linspace))
'''
'''
#0表示10的零次方,2表示10的二次方,5表示生成5个元素
ArrayLog=logspace(0,2,5)
print(ArrayLog)

#生成随机的数组
ArrayRandom=random.random((2,2,4))
print(ArrayRandom)

#astype可以更改数组的类型
ndarray=arange(20)
print(ndarray.dtype)
ndarray2=ndarray.astype(int64)
print(ndarray2.dtype)
'''


'''
StringArray=array(
    [
        "python12","java","golang","c#"
    ],dtype='S8'
)
print(StringArray.dtype)


arr1=array([1,2,3,4,5])
print(arr1+2)
print(1/arr1)
print(arr1**2)


arr1=array([[1,2,3],[3,2,1]])
arr2=array([[1.2,2.3,3.1],[1,3,4]])

print(arr1+arr2)
print(arr1/arr2)
print(arr2.dtype)



arr1=array\
    (
        [
            [1,3,2,6],
            [2,3,4,5],
            [4,5,2,6]
        ]
    )

arr2=array\
    (
        [
            [1, 3, 2, 6],
            [2, 3, 4, 5],
            [4, 5, 2, 6],
            [4, 5, 2, 6]
        ]
    )
print(arr1.dot(arr2))
print(dot(arr1,arr2))


arr2=array\
    (
        [
            [
                [1, 3, 2, 6],
                [2, 3, 4, 5],
                [4, 5, 2, 6],
                [4, 5, 2, 6]
            ],

            [
                [1, 3, 2, 6],
                [2, 3, 4, 5],
                [4, 5, 2, 6],
                [4, 5, 2, 6]
            ]
        ]
    )
print(arr2[1][0][1:3])
print()
print(arr2[0,0:2,1:3])

A=random.random((4,4))
print(A)
print(A<0.5)
#True位置上的元素取出组成一个新的数组
print(A[A<0.5])


names=array(['Tom','Merry','carry'])
scores=array\
    (
        [
            [1,2,3,4],
            [7,1,2,4],
            [2, 3, 1, 4]
        ]
    )
print(scores[~((names=='Tom')|(names=='carry'))])
print(scores[(names!='Tom')&(names!='carry')])

#8行四列
arr=arange(32).reshape(8,4)
print(arr.transpose())
print(arr.T)


#计算方差
arr=array([1.0,2.0,3.0,4.0])
print(arr.std())
print(sqrt(pow(arr-arr.mean(),2).sum()/arr.size))

arr=arange(32).reshape(8,4)
print(arr)
print(arr.max(axis=1))


arr1=arange(1,5)
arr2=arange(2,6)
print(arr1)
print(arr2)

condition=array([True,False,True,False])
print(where(condition,arr1,arr2))
result=[(x if c else y)for x,y,c in zip(arr1,arr2,condition)]
print(result)

x=y=z=arange(0.0,5.0,1.0)
#savetxt('test.out',x,delimiter=',')
#savetxt('test2.out',(x,y,z),delimiter=',')
#savetxt('test3.out',x,fmt='%1.4e')
x,y,z=loadtxt('test2.out',delimiter=',')
print(str(x)+str(y))



data=array\
    (
        [
            [2017,2018,2019,2020],
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,13]
        ],dtype=float64
    )
savetxt('data2.csv',data,delimiter=',',fmt="%d")

#data=arange(0,16).reshape(4,4)
#savetxt('data.csv',data,delimiter=',')
data2=genfromtxt('data2.csv',delimiter=',',names=True)
print(data2.dtype)
print(data2['2017'])
'''

c={'A':1,'B':0,'B':2,'A':3}
d=sorted(c.items(),key=operator.itemgetter(1),reverse=True)
print(d)
