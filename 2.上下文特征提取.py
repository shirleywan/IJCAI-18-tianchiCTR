import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 获取训练集一中上下文相关特征
train1_f = pd.read_csv('newdata/1train_cut.csv')
tra = train1_f
# 展示在该页的被点击的次数
t = tra[['context_page_id']]#提取context_page_id
t['page_click']=1
t = t.groupby('context_page_id').agg('sum').reset_index()#t用来求sum，注意写法！
tra = pd.merge(tra,t,on='context_page_id',how='left') #merge  vs concat:当axis=0时，是行的append；axis=1时，相当于全联接
# 展示在该页被购买的次数
t = tra[['context_page_id','is_trade']]
t = t.groupby('context_page_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'page_buy'})
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页的购买率
tra['page_buy_rate'] = tra['page_buy']/tra['page_click']#context的购买率/点击率
# 展示在该页



tra[['page_click','page_buy','page_buy_rate','context_page_id']].to_csv('newdata/context_feature1.csv',index=None)
exit(0)

# 获取训练集二中上下文相关特征
train1_f = pd.read_csv('data/train2_f.csv')
tra = train1_f
# 展示在该页的被点击的次数
t = tra[['context_page_id']]
t['page_click']=1
t = t.groupby('context_page_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页被购买的次数
t = tra[['context_page_id','is_trade']]
t = t.groupby('context_page_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'page_buy'})
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页的购买率
tra['page_buy_rate'] = tra['page_buy']/tra['page_click']



tra[['page_click','page_buy','page_buy_rate','context_page_id']].to_csv('data/context_feature2.csv',index=None)


# 获取训练集三中上下文相关特征
train1_f = pd.read_csv('data/train3_f.csv')
tra = train1_f
# 展示在该页的被点击的次数
t = tra[['context_page_id']]
t['page_click']=1
t = t.groupby('context_page_id').agg('sum').reset_index()
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页被购买的次数
t = tra[['context_page_id','is_trade']]
t = t.groupby('context_page_id').agg('sum').reset_index()
t = t.rename(columns={'is_trade':'page_buy'})
tra = pd.merge(tra,t,on='context_page_id',how='left')
# 展示在该页的购买率
tra['page_buy_rate'] = tra['page_buy']/tra['page_click']
# 展示在该页



tra[['page_click','page_buy','page_buy_rate','context_page_id']].to_csv('data/context_feature3.csv',index=None)
