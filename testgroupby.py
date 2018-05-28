import pandas as pd
import numpy as np

df = pd.DataFrame({'key1':list('aabba'),
                  'key2': ['one','two','one','two','one'],
                  'data1':[-0.233405,-0.232103,0.200875,-1.437782,1.056224],
                  'data2':[-0.756316,-0.095864,0.598282,0.107547,0.736629]})
# print(df)

# for name, group in df.groupby('key1'):
#         print (name)
#         print (group)

# piece=dict(list(df.groupby('key1')))
# print(piece['a'])

# data1_dic=list(df.groupby('key1')['data1'])
# print(data1_dic)
print(df.groupby(['key1','key2'])[['data2']].mean())
print('-----------groupby test-------------')

people=pd.DataFrame(#np.random.randn(5,5), #随机生成5*5矩阵
                   [[0.788276,-2.232843,1.956668,2.946316,0.043517],
                    [-0.688506,-0.187575,-0.048742,1.491272,-0.636704],
                    [-0.133487,0,0,-0.461242,-0.248194],
                    [-1.963499,-0.120511,-0.371084,0.423286,-1.062485],
                    [0.110028,-0.932493,1.343791,-1.928363,-0.364745]],
                   columns=list('abcde'),
                   index=['Joe','Steve','Wes','Jim','Travis'])

people.ix[2:3,['b','c']]=np.nan #设置几个nan
# print(people)
mapping={'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}
# by_column=people.groupby(mapping,axis=1)
# by_column.sum()
# print(list(by_column))

# map_series=pd.Series(mapping)
# print(map_series)
# print(people.groupby(map_series,axis=1).count()) #这里用来数每个人有几个red和blue，中间有问题的只有nan项

# print(list(people.groupby(len)))#这里按人名的长度进行分组，仅仅传入 len 即可
# print(people.groupby(len).sum())

# key_list=['one','one','one','two','two']
# print(people.groupby([len,key_list]).sum())

# columns=pd.MultiIndex.from_arrays([['US','US','US','JP','JP'],[1,3,5,1,3]],
#      names=['cty','tenor'])
# hier_df=pd.DataFrame(np.random.randn(4,5),columns=columns)
# print(hier_df)

# def peak_to_peak(arr):
#     return arr.max()-arr.min()
#
# print(people.groupby(mapping,axis=1).agg('sum')) #按列分组后做sum
# print(people.groupby(mapping,axis=1).agg('count'))
# print(people.groupby(mapping,axis=1).agg('median'))
# print(people.groupby(mapping,axis=1).agg('first'))
# print(people.groupby(mapping,axis=1).agg('last'))

# key=['one','two','one','two','one']
# print(people.groupby(key).mean())
# def demean(arr):
#     return arr-arr.mean()

# demeaned=people.groupby(key).transform(demean) #减去平均值
# print(demeaned)
# print(demeaned.groupby(key).mean())

frame=pd.DataFrame({'data1':np.random.randn(10),
                   'data2': np.random.randn(10)})
print(frame[:10])
factor=pd.cut(frame.data1,4) #划分为4个区间
print(factor[:10]) #这里输出data1中每个值属于的区间