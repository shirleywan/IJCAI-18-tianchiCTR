import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 从训练集一中获取用户店铺交互特征
train1_f = pd.read_csv('newdata/1train_cut.csv')
shop_fea = pd.read_csv('newdata/1/shop_feature1.csv')
shop_fea.drop_duplicates(inplace=True)
user_fea = pd.read_csv('newdata/1/user_feature1.csv')
user_fea.drop_duplicates(inplace=True)
train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
user_shop = train1_f
# 该用户点击该店铺次数
t = user_shop[['user_id','shop_id']]
t['user_shop_click_total'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')

# 用户购买该店铺次数
t = user_shop[['user_id','shop_id','is_trade']]
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# 用户购买该店铺率
user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']

# 用户购买该店铺占店铺售卖比重
user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']

# 用户购买该店铺占用户购买比重
user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']

user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('newdata/1/user_shop_feature1.csv',index=None)

# 该星级用户在该店铺点击数目
t = user_shop[['user_star_level','shop_id']]
t['star_shop_click'] = 1
t = t.groupby(['user_star_level','shop_id']).agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_star_level','shop_id'],how='left')
# 购买
t = user_shop[['user_star_level','shop_id','is_trade']]
t = t.groupby(['user_star_level','shop_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_shop_buy'})
user_shop = pd.merge(user_shop,t,on=['user_star_level','shop_id'],how='left')
# lv
user_shop['star_shop_rate'] = user_shop['star_shop_buy']/user_shop['star_shop_click']
user_shop[['star_shop_click','star_shop_buy','user_star_level','shop_id']].to_csv('newdata/1/star_shop_feature1.csv',index=None)
# 该lei用户在该店铺点击数目
t = user_shop[['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']]
t['gosa_shop_click'] = 1
t = t.groupby(['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']).agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id'],how='left')
# 购买
t = user_shop[['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id','is_trade']]
t = t.groupby(['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'gosa_shop_buy'})
user_shop = pd.merge(user_shop,t,on=['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id'],how='left')
# lv
user_shop['gosa_shop_rate'] = user_shop['gosa_shop_buy']/user_shop['gosa_shop_click']
user_shop[['gosa_shop_click','gosa_shop_buy','user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']].to_csv('newdata/1/gosa_shop_feature1.csv',index=None)


exit(0)

# 从训练集二中获取用户店铺交互特征
train1_f = pd.read_csv('data/train2_f.csv')
shop_fea = pd.read_csv('data/shop_feature2.csv')
shop_fea.drop_duplicates(inplace=True)
user_fea = pd.read_csv('data/user_feature2.csv')
user_fea.drop_duplicates(inplace=True)
train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
user_shop = train1_f
# 该用户点击该店铺次数
t = user_shop[['user_id','shop_id']]
t['user_shop_click_total'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')

# 用户购买该店铺次数
t = user_shop[['user_id','shop_id','is_trade']]
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# 用户购买该店铺率
user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']

# 用户购买该店铺占店铺售卖比重
user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']

# 用户购买该店铺占用户购买比重
user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']

user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('data/user_shop_feature2.csv',index=None)
# 该星级用户在该店铺点击数目
t = user_shop[['user_star_level','shop_id']]
t['star_shop_click'] = 1
t = t.groupby(['user_star_level','shop_id']).agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_star_level','shop_id'],how='left')
# 购买
t = user_shop[['user_star_level','shop_id','is_trade']]
t = t.groupby(['user_star_level','shop_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_shop_buy'})
user_shop = pd.merge(user_shop,t,on=['user_star_level','shop_id'],how='left')
# lv
user_shop['star_shop_rate'] = user_shop['star_shop_buy']/user_shop['star_shop_click']
user_shop[['star_shop_click','star_shop_buy','user_star_level','shop_id']].to_csv('data/star_shop_feature2.csv',index=None)
# 该lei用户在该店铺点击数目
t = user_shop[['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']]
t['gosa_shop_click'] = 1
t = t.groupby(['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']).agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id'],how='left')
# 购买
t = user_shop[['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id','is_trade']]
t = t.groupby(['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'gosa_shop_buy'})
user_shop = pd.merge(user_shop,t,on=['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id'],how='left')
# lv
user_shop['gosa_shop_rate'] = user_shop['gosa_shop_buy']/user_shop['gosa_shop_click']
user_shop[['gosa_shop_click','gosa_shop_buy','user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']].to_csv('data/gosa_shop_feature2.csv',index=None)


# 从训练集三中获取用户店铺交互特征
train1_f = pd.read_csv('data/train3_f.csv')
shop_fea = pd.read_csv('data/shop_feature3.csv')
shop_fea.drop_duplicates(inplace=True)
user_fea = pd.read_csv('data/user_feature3.csv')
user_fea.drop_duplicates(inplace=True)
train1_f = pd.merge(train1_f,shop_fea,on='shop_id',how='left')
train1_f = pd.merge(train1_f,user_fea,on='user_id',how='left')
user_shop = train1_f
# 该用户点击该店铺次数
t = user_shop[['user_id','shop_id']]
t['user_shop_click_total'] = 1
t = t.groupby(['user_id','shop_id'])['user_shop_click_total'].agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')

# 用户购买该店铺次数
t = user_shop[['user_id','shop_id','is_trade']]
t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
t.rename(columns={'is_trade':'user_shop_click_buy_total'},inplace=True)
user_shop = pd.merge(user_shop,t,on=['user_id','shop_id'],how='left')
# 用户购买该店铺率
user_shop['user_shop_click_buy_rate'] = user_shop['user_shop_click_buy_total']/user_shop['user_shop_click_total']

# 用户购买该店铺占店铺售卖比重
user_shop['user_shop_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['shop_click_buy_total']

# 用户购买该店铺占用户购买比重
user_shop['shop_user_click_buy_rate']=user_shop['user_shop_click_buy_total']/user_shop['user_click_buy_total']

user_shop[['user_shop_click_buy_rate','user_shop_click_buy_total','user_shop_click_total','user_shop_click_buy_rate','shop_user_click_buy_rate','user_id','shop_id']].to_csv('data/user_shop_feature3.csv',index=None)
# 该星级用户在该店铺点击数目
t = user_shop[['user_star_level','shop_id']]
t['star_shop_click'] = 1
t = t.groupby(['user_star_level','shop_id']).agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_star_level','shop_id'],how='left')
# 购买
t = user_shop[['user_star_level','shop_id','is_trade']]
t = t.groupby(['user_star_level','shop_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'star_shop_buy'})
user_shop = pd.merge(user_shop,t,on=['user_star_level','shop_id'],how='left')
# lv
user_shop['star_shop_rate'] = user_shop['star_shop_buy']/user_shop['star_shop_click']
user_shop[['star_shop_click','star_shop_buy','user_star_level','shop_id']].to_csv('data/star_shop_feature3.csv',index=None)

# 该lei用户在该店铺点击数目
t = user_shop[['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']]
t['gosa_shop_click'] = 1
t = t.groupby(['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']).agg('sum').reset_index()
user_shop = pd.merge(user_shop,t,on=['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id'],how='left')
# 购买
t = user_shop[['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id','is_trade']]
t = t.groupby(['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']).agg('sum').reset_index()
t = t.rename(columns={'is_trade':'gosa_shop_buy'})
user_shop = pd.merge(user_shop,t,on=['user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id'],how='left')
# lv
user_shop['gosa_shop_rate'] = user_shop['gosa_shop_buy']/user_shop['gosa_shop_click']
user_shop[['gosa_shop_click','gosa_shop_buy','user_star_level','user_gender_id','user_occupation_id','user_age_level','shop_id']].to_csv('data/gosa_shop_feature3.csv',index=None)
