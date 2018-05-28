import pandas as pd
import xgboost as xgb
from sklearn.ensemble import  GradientBoostingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def hasTrade(trdata,tedata):
    pdata = trdata[["is_trade","first_cate","user_id"]].drop_duplicates(inplace=False)
    pdata = (pdata.loc[pdata["is_trade"]==1])[["first_cate","user_id"]] #只返回满足条件的first_cate和user_id
    pdata["has_trade"] = 1 #使这时的has_trade为1
    pdata = pdata[["first_cate","user_id","has_trade"]] #当前pdata有3列
    tedata = pd.merge(tedata, pdata, 'left')#将原数据和新做出来的数据结合起来，tedata放在left
    tedata.loc[tedata["has_trade"]!=1,"has_trade"]=0 #has_trade一项不为1的设置为0
    trdata = pd.merge(trdata, pdata, 'left')
    trdata.loc[trdata["has_trade"]!=1,"has_trade"]=0
    return trdata,tedata

def cateHit(trdata,tedata,cateName): #使用second-cate
    pdata = trdata[[cateName,"user_id"]].groupby([cateName,"user_id"]).size().reset_index().rename(columns={0:"cate_hit"})
        #返回：second_cate，user_id，cate_hit；三个选项，cate_hit都为1
    tedata = pd.merge(tedata, pdata, 'left')
    tedata.fillna(0) #填补缺失值
    #tedata.loc[tedata["cate_hit"]!=1,"cate_hit"]=0
    trdata = pd.merge(trdata, pdata, 'left')
    trdata.fillna(0)
    #trdata.loc[trdata["cate_hit"]!=1,"cate_hit"]=0
    return trdata,tedata


def processHasTrade(test,labelName):
    test.loc[test["has_trade"]==1,[labelName]] *= 1.1
    test.loc[test["has_trade"]==0,[labelName]] *= 0.9


train1_p = pd.read_csv('newdata/1train_cut.csv')

cate = pd.get_dummies(train1_p['cate2'])
train1_p = pd.concat([train1_p,cate],axis=1)

user_occupation_id = pd.get_dummies(train1_p['user_occupation_id']) #可以将类别变量转换成新增的虚拟变量 / 指示变量。
train1_p = pd.concat([train1_p,user_occupation_id],axis=1)

user_feature1 = pd.read_csv('newdata/1/user_feature1.csv')
user_feature1.drop_duplicates(inplace=True)
shop_feature1 = pd.read_csv('newdata/1/shop_feature1.csv')
shop_feature1.drop_duplicates(inplace=True)
item_feature1 = pd.read_csv('newdata/1/item_feature1.csv')
item_feature1.drop_duplicates(subset=['item_id'],inplace=True)#
context_feature1 = pd.read_csv('newdata/1/context_feature1.csv')
context_feature1.drop_duplicates(inplace=True)
user_shop_feature1 = pd.read_csv('newdata/1/user_shop_feature1.csv')
user_shop_feature1.drop_duplicates(inplace=True)
user_item_feature1 = pd.read_csv('newdata/1/user_item_feature1.csv')
user_item_feature1.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
user_brand_feature1 = pd.read_csv('newdata/1/user_brand_feature1.csv')
user_brand_feature1.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
user_context_feature1 = pd.read_csv('newdata/1/user_context_feature1.csv')
user_context_feature1.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
user_price_feature1 = pd.read_csv('newdata/1/user_price_feature1.csv')
user_price_feature1.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
user_collected_feature1 = pd.read_csv('newdata/1/user_collected_feature1.csv')
user_collected_feature1.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
user_pv_feature1 = pd.read_csv('newdata/1/user_pv_feature1.csv')
user_pv_feature1.drop_duplicates(subset=['user_id','item_pv_level'],inplace=True)
user_hour_feature1 = pd.read_csv('newdata/1/user_hour_feature1.csv')
user_hour_feature1.drop_duplicates(subset=['user_id','hour'],inplace=True)
user_cate_feature1 = pd.read_csv('newdata/1/user_cate_feature1.csv')
user_cate_feature1.drop_duplicates(subset=['user_id','cate2'],inplace=True)
user_city_feature1 = pd.read_csv('newdata/1/user_city_feature1.csv')
user_city_feature1.drop_duplicates(subset=['user_id','item_city_id'],inplace=True)
shop_brand_feature1 = pd.read_csv('newdata/1/shop_brand_feature1.csv')
shop_brand_feature1.drop_duplicates(subset=['shop_id','item_brand_id'],inplace=True)
shop_item_feature1 = pd.read_csv('data/shop_item_feature1.csv')
shop_item_feature1.drop_duplicates(inplace=True)
shop_hour_feature1 = pd.read_csv('data/shop_hour_feature1.csv')
shop_hour_feature1.drop_duplicates(inplace=True)
user_shop_item_feature1 = pd.read_csv('data/user_shop_item_feature1.csv')
user_shop_item_feature1.drop_duplicates(subset=['user_id','shop_id','item_id'],inplace=True)
age_item_feature1 = pd.read_csv('data/age_item_feature1.csv')
age_item_feature1.drop_duplicates(subset=['user_age_level','item_id'],inplace=True)
star_item_feature1 = pd.read_csv('data/star_item_feature1.csv')
star_item_feature1.drop_duplicates(subset=['user_star_level','item_id'],inplace=True)
occupation_item_feature1 = pd.read_csv('data/occupation_item_feature1.csv')
occupation_item_feature1.drop_duplicates(subset=['user_occupation_id','item_id'],inplace=True)
age_occupation_item_feature1 = pd.read_csv('data/age_occupation_item_feature1.csv')
age_occupation_item_feature1.drop_duplicates(subset=['user_age_level','user_occupation_id','item_id'],inplace=True)
age_star_item_feature1 = pd.read_csv('data/age_star_item_feature1.csv')
age_star_item_feature1.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
occupation_star_item_feature1 = pd.read_csv('data/occupation_star_item_feature1.csv')
occupation_star_item_feature1.drop_duplicates(subset=['user_occupation_id','user_star_level','item_id'],inplace=True)
occupation_star_age_item_feature1 = pd.read_csv('data/occupation_star_age_item_feature1.csv')
occupation_star_age_item_feature1.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_id'],inplace=True)
age_cate_feature1 = pd.read_csv('data/age_cate_feature1.csv')
age_cate_feature1.drop_duplicates(subset=['user_age_level','cate2'],inplace=True)
star_cate_feature1 = pd.read_csv('data/star_cate_feature1.csv')
star_cate_feature1.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
occupation_cate_feature1 = pd.read_csv('data/occupation_cate_feature1.csv')
occupation_cate_feature1.drop_duplicates(subset=['user_occupation_id','cate2'],inplace=True)
item_brand_feature1 = pd.read_csv('data/item_brand_feature1.csv')
item_brand_feature1.drop_duplicates(subset=['item_id','item_brand_id'],inplace=True)
occupation_brand_feature1 = pd.read_csv('data/occupation_brand_feature1.csv')
occupation_brand_feature1.drop_duplicates(subset=['user_occupation_id','item_brand_id'],inplace=True)
star_brand_feature1 = pd.read_csv('data/star_brand_feature1.csv')
star_brand_feature1.drop_duplicates(subset=['user_star_level','item_brand_id'],inplace=True)
occupation_star_age_brand_feature1 = pd.read_csv('data/occupation_star_age_brand_feature1.csv')
occupation_star_age_brand_feature1.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_brand_id'],inplace=True)
star_price_feature1 = pd.read_csv('data/star_price_feature1.csv')
star_price_feature1.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
occupation_star_age_price_feature1 = pd.read_csv('data/occupation_star_age_price_feature1.csv')
occupation_star_age_price_feature1.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_price_level'],inplace=True)
star_shop_feature1 = pd.read_csv('data/star_shop_feature2.csv')
star_shop_feature1.drop_duplicates(subset=['user_star_level','shop_id'],inplace=True)
gosa_feature1 = pd.read_csv('data/gosa_feature1.csv')
gosa_feature1.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level'],inplace=True)
gosa_shop_feature1 = pd.read_csv('data/gosa_shop_feature1.csv')
gosa_shop_feature1.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level','shop_id'],inplace=True)
gosa_cate_feature1 = pd.read_csv('data/gosa_cate_feature1.csv')
gosa_cate_feature1.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level','cate2'],inplace=True)
star_age_cate_feature1 = pd.read_csv('data/star_age_cate_feature1.csv')
star_age_cate_feature1.drop_duplicates(subset=['user_star_level','user_age_level','cate2'],inplace=True)

other_user_feature1 = pd.read_csv('data/other_user_feature1.csv')
other_user_feature1.drop_duplicates(inplace=True)
other_item_feature1 = pd.read_csv('data/other_item_feature1.csv')
other_item_feature1.drop_duplicates(inplace=True)
other_shop_feature1 = pd.read_csv('data/other_shop_feature1.csv')
other_shop_feature1.drop_duplicates(inplace=True)
other_brand_feature1 = pd.read_csv('data/other_brand_feature1.csv')
other_brand_feature1.drop_duplicates(inplace=True)
other_user_item_feature1 = pd.read_csv('data/other_user_item_feature1.csv')
other_user_item_feature1.drop_duplicates(inplace=True)
other_user_shop_feature1 = pd.read_csv('data/other_user_shop_feature1.csv')
other_user_shop_feature1.drop_duplicates(inplace=True)
other_user_hour_feature1 = pd.read_csv('data/other_user_hour_feature1.csv')
other_user_hour_feature1.drop_duplicates(inplace=True)
other_user_cate_feature1 = pd.read_csv('data/other_user_cate_feature1.csv')
other_user_cate_feature1.drop_duplicates(inplace=True)
other_item_shop_feature1 = pd.read_csv('data/other_item_shop_feature1.csv')
other_item_shop_feature1.drop_duplicates(inplace=True)
other_user_brand_feature1 = pd.read_csv('data/other_user_brand_feature1.csv')
other_user_brand_feature1.drop_duplicates(inplace=True)
# other_shop_brand_feature1 = pd.read_csv('data/other_shop_brand_feature1.csv')
# other_shop_brand_feature1.drop_duplicates(inplace=True)
other_user_shop_item_feature1 = pd.read_csv('data/other_user_shop_item_feature1.csv')
other_user_shop_item_feature1.drop_duplicates(inplace=True)
other_user_item_hour_feature1 = pd.read_csv('data/other_user_item_hour_feature1.csv')
other_user_item_hour_feature1.drop_duplicates(inplace=True)
other_user_price_feature1 = pd.read_csv('data/other_user_price_feature1.csv')
other_user_price_feature1.drop_duplicates(inplace=True)
other_user_collected_feature1 = pd.read_csv('data/other_user_collected_feature1.csv')
other_user_collected_feature1.drop_duplicates(inplace=True)
other_user_sales_feature1 = pd.read_csv('data/other_user_sales_feature1.csv')
other_user_sales_feature1.drop_duplicates(inplace=True)
other_shop_hour_feature1 = pd.read_csv('data/other_shop_hour_feature1.csv')
other_shop_hour_feature1.drop_duplicates(inplace=True)
print(train1_p.shape) #第一个输出：(63614, 50),(999999, 89)

train1_p = pd.merge(train1_p,user_feature1,on='user_id',how='left')
train1_p = pd.merge(train1_p,shop_feature1,on='shop_id',how='left')
train1_p = pd.merge(train1_p,item_feature1,on='item_id',how='left')
train1_p = pd.merge(train1_p,context_feature1,on='context_page_id',how='left')
train1_p = pd.merge(train1_p,user_shop_feature1,on=['user_id','shop_id'],how='left')
train1_p = pd.merge(train1_p,user_item_feature1,on=['user_id','item_id'],how='left')
train1_p = pd.merge(train1_p,user_brand_feature1,on=['user_id','item_brand_id'],how='left')
train1_p = pd.merge(train1_p,user_price_feature1,on=['user_id','item_price_level'],how='left')
train1_p = pd.merge(train1_p,user_collected_feature1,on=['user_id','item_collected_level'],how='left')
# train1_p = pd.merge(train1_p,user_pv_feature1,on=['user_id','item_pv_level'],how='left')
train1_p = pd.merge(train1_p,user_context_feature1,on=['user_id','context_page_id'],how='left')
# train1_p = pd.merge(train1_p,user_hour_feature1,on=['user_id','hour'],how='left')
train1_p = pd.merge(train1_p,user_cate_feature1,on=['user_id','cate2'],how='left')
train1_p = pd.merge(train1_p,shop_item_feature1,on=['shop_id','item_id'],how='left')
# train1_p = pd.merge(train1_p,shop_hour_feature1,on=['shop_id','hour'],how='left')
train1_p = pd.merge(train1_p,user_shop_item_feature1,on=['user_id','shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_feature1,on=['user_id'],how='left')
train1_p = pd.merge(train1_p,other_item_feature1,on=['item_id'],how='left')
train1_p = pd.merge(train1_p,other_shop_feature1,on=['shop_id'],how='left')
train1_p = pd.merge(train1_p,other_brand_feature1,on=['item_brand_id'],how='left')
train1_p = pd.merge(train1_p,other_user_item_feature1,on=['user_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_shop_feature1,on=['user_id','shop_id'],how='left')
train1_p = pd.merge(train1_p,other_user_hour_feature1,on=['user_id','hour'],how='left')
train1_p = pd.merge(train1_p,other_item_shop_feature1,on=['shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_brand_feature1,on=['user_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,other_shop_brand_feature1,on=['shop_id','item_brand_id'],how='left')
train1_p = pd.merge(train1_p,other_user_shop_item_feature1,on=['user_id','shop_id','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_item_hour_feature1,on=['user_id','hour','item_id'],how='left')
train1_p = pd.merge(train1_p,other_user_price_feature1,on=['user_id','item_price_level'],how='left')
train1_p = pd.merge(train1_p,other_user_collected_feature1,on=['user_id','item_collected_level'],how='left')
# train1_p = pd.merge(train1_p,other_user_sales_feature1,on=['user_id','item_sales_level'],how='left')
train1_p = pd.merge(train1_p,other_user_cate_feature1,on=['user_id','cate2'],how='left')
train1_p = pd.merge(train1_p,age_item_feature1,on=['user_age_level','item_id'],how='left')
train1_p = pd.merge(train1_p,star_item_feature1,on=['user_star_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,star_shop_feature1,on=['user_star_level','shop_id'],how='left')

# train1_p = pd.merge(train1_p,occupation_item_feature1,on=['user_occupation_id','item_id'],how='left')
train1_p = pd.merge(train1_p,age_star_item_feature1,on=['user_age_level','user_star_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_occupation_item_feature1,on=['user_age_level','user_occupation_id','item_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_star_item_feature1,on=['user_occupation_id','user_star_level','item_id'],how='left')
train1_p = pd.merge(train1_p,occupation_star_age_item_feature1,on=['user_occupation_id','user_star_level','user_age_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_star_age_brand_feature1,on=['user_occupation_id','user_star_level','user_age_level','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_star_age_price_feature1,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')

# train1_p = pd.merge(train1_p,age_cate_feature1,on=['user_age_level','cate2'],how='left')
train1_p = pd.merge(train1_p,star_cate_feature1,on=['user_star_level','cate2'],how='left')
# train1_p = pd.merge(train1_p,star_age_cate_feature1,on=['user_star_level','user_age_level','cate2'],how='left')
train1_p = pd.merge(train1_p,star_price_feature1,on=['user_star_level','item_price_level'],how='left')
# train1_p = pd.merge(train1_p,)
# train1_p = pd.merge(train1_p,star_shop_feature1,on=['user_star_level','shop_id'],how='left')
# train1_p = pd.merge(train1_p,star_brand_feature1,on=['user_star_level','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_cate_feature1,on=['user_occupation_id','cate2'],how='left')
# train1_p = pd.merge(train1_p,item_brand_feature1,on=['item_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,occupation_brand_feature1,on=['user_occupation_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,star_brand_feature1,on=['item_brand_id','user_star_level'],how='left')
# train1_p = pd.merge(train1_p,gosa_feature1,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level'],how='left')
# train1_p = pd.merge(train1_p,gosa_cate_feature1,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level','cate2'],how='left')
# train1_p = pd.merge(train1_p,gosa_shop_feature1,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level','shop_id'],how='left')

print(train1_p.shape) #(63614, 176),(999999, 215)
train1_p.to_csv('newdata/newtrain1.csv', index=None)
exit(0)

'''
train2_p = pd.read_csv('data/train2_p.csv')
# train2_ceshi = pd.read_csv('data/train2_p.csv')

cate = pd.get_dummies(train2_p['cate2'])
train2_p = pd.concat([train2_p,cate],axis=1)

user_occupation_id = pd.get_dummies(train2_p['user_occupation_id'])
train2_p = pd.concat([train2_p,user_occupation_id],axis=1)



user_feature2 = pd.read_csv('data/user_feature2.csv')
user_feature2.drop_duplicates(inplace=True)
shop_feature2 = pd.read_csv('data/shop_feature2.csv')
shop_feature2.drop_duplicates(inplace=True)
item_feature2 = pd.read_csv('data/item_feature2.csv')
item_feature2.drop_duplicates(subset=['item_id'],inplace=True)#
context_feature2 = pd.read_csv('data/context_feature2.csv')
context_feature2.drop_duplicates(inplace=True)
user_shop_feature2 = pd.read_csv('data/user_shop_feature2.csv')
user_shop_feature2.drop_duplicates(inplace=True)
user_item_feature2 = pd.read_csv('data/user_item_feature2.csv')
user_item_feature2.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
user_brand_feature2 = pd.read_csv('data/user_brand_feature2.csv')
user_brand_feature2.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
user_context_feature2 = pd.read_csv('data/user_context_feature2.csv')
user_context_feature2.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
user_price_feature2 = pd.read_csv('data/user_price_feature2.csv')
user_price_feature2.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
user_collected_feature2 = pd.read_csv('data/user_collected_feature2.csv')
user_collected_feature2.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
user_pv_feature2 = pd.read_csv('data/user_pv_feature2.csv')
user_pv_feature2.drop_duplicates(subset=['user_id','item_pv_level'],inplace=True)
user_hour_feature2 = pd.read_csv('data/user_hour_feature2.csv')
user_hour_feature2.drop_duplicates(subset=['user_id','hour'],inplace=True)
user_cate_feature2 = pd.read_csv('data/user_cate_feature2.csv')
user_cate_feature2.drop_duplicates(subset=['user_id','cate2'],inplace=True)
user_city_feature2 = pd.read_csv('data/user_city_feature2.csv')
user_city_feature2.drop_duplicates(subset=['user_id','item_city_id'],inplace=True)
shop_brand_feature2 = pd.read_csv('data/shop_brand_feature2.csv')
shop_brand_feature2.drop_duplicates(subset=['shop_id','item_brand_id'],inplace=True)
shop_item_feature2 = pd.read_csv('data/shop_item_feature2.csv')
shop_item_feature2.drop_duplicates(inplace=True)
shop_hour_feature2 = pd.read_csv('data/shop_hour_feature2.csv')
shop_hour_feature2.drop_duplicates(inplace=True)
user_shop_item_feature2 = pd.read_csv('data/user_shop_item_feature2.csv')
user_shop_item_feature2.drop_duplicates(inplace=True)
age_item_feature2 = pd.read_csv('data/age_item_feature2.csv')
age_item_feature2.drop_duplicates(subset=['user_age_level','item_id'],inplace=True)
star_item_feature2 = pd.read_csv('data/star_item_feature2.csv')
star_item_feature2.drop_duplicates(subset=['user_star_level','item_id'],inplace=True)
occupation_item_feature2 = pd.read_csv('data/occupation_item_feature2.csv')
occupation_item_feature2.drop_duplicates(subset=['user_occupation_id','item_id'],inplace=True)
age_occupation_item_feature2 = pd.read_csv('data/age_occupation_item_feature2.csv')
age_occupation_item_feature2.drop_duplicates(subset=['user_age_level','user_occupation_id','item_id'],inplace=True)
age_star_item_feature2 = pd.read_csv('data/age_star_item_feature2.csv')
age_star_item_feature2.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
occupation_star_item_feature2 = pd.read_csv('data/occupation_star_item_feature2.csv')
occupation_star_item_feature2.drop_duplicates(subset=['user_occupation_id','user_star_level','item_id'],inplace=True)
occupation_star_age_item_feature2 = pd.read_csv('data/occupation_star_age_item_feature2.csv')
occupation_star_age_item_feature2.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_id'],inplace=True)
age_cate_feature2 = pd.read_csv('data/age_cate_feature2.csv')
age_cate_feature2.drop_duplicates(subset=['user_age_level','cate2'],inplace=True)
star_cate_feature2 = pd.read_csv('data/star_cate_feature2.csv')
star_cate_feature2.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
occupation_cate_feature2 = pd.read_csv('data/occupation_cate_feature2.csv')
occupation_cate_feature2.drop_duplicates(subset=['user_occupation_id','cate2'],inplace=True)
item_brand_feature2 = pd.read_csv('data/item_brand_feature2.csv')
item_brand_feature2.drop_duplicates(subset=['item_id','item_brand_id'],inplace=True)
occupation_brand_feature2 = pd.read_csv('data/occupation_brand_feature2.csv')
occupation_brand_feature2.drop_duplicates(subset=['user_occupation_id','item_brand_id'],inplace=True)
star_brand_feature2 = pd.read_csv('data/star_brand_feature2.csv')
star_brand_feature2.drop_duplicates(subset=['user_star_level','item_brand_id'],inplace=True)
occupation_star_age_brand_feature2 = pd.read_csv('data/occupation_star_age_brand_feature2.csv')
occupation_star_age_brand_feature2.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_brand_id'],inplace=True)
occupation_star_age_price_feature2 = pd.read_csv('data/occupation_star_age_price_feature2.csv')
occupation_star_age_price_feature2.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_price_level'],inplace=True)
star_price_feature2 = pd.read_csv('data/star_price_feature2.csv')
star_price_feature2.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
star_shop_feature2 = pd.read_csv('data/star_shop_feature2.csv')
star_shop_feature2.drop_duplicates(subset=['user_star_level','shop_id'],inplace=True)
gosa_feature2 = pd.read_csv('data/gosa_feature2.csv')
gosa_feature2.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level'],inplace=True)
gosa_shop_feature2 = pd.read_csv('data/gosa_shop_feature2.csv')
gosa_shop_feature2.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level','shop_id'],inplace=True)
gosa_cate_feature2 = pd.read_csv('data/gosa_cate_feature2.csv')
gosa_cate_feature2.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level','cate2'],inplace=True)
star_age_cate_feature2 = pd.read_csv('data/star_age_cate_feature2.csv')
star_age_cate_feature2.drop_duplicates(subset=['user_star_level','user_age_level','cate2'],inplace=True)

other_user_feature2 = pd.read_csv('data/other_user_feature2.csv')
other_user_feature2.drop_duplicates(inplace=True)
other_item_feature2 = pd.read_csv('data/other_item_feature2.csv')
other_item_feature2.drop_duplicates(inplace=True)
other_shop_feature2 = pd.read_csv('data/other_shop_feature2.csv')
other_shop_feature2.drop_duplicates(inplace=True)
other_brand_feature2 = pd.read_csv('data/other_brand_feature2.csv')
other_brand_feature2.drop_duplicates(inplace=True)
other_user_item_feature2 = pd.read_csv('data/other_user_item_feature2.csv')
other_user_item_feature2.drop_duplicates(inplace=True)
other_user_shop_feature2 = pd.read_csv('data/other_user_shop_feature2.csv')
other_user_shop_feature2.drop_duplicates(inplace=True)
other_user_hour_feature2 = pd.read_csv('data/other_user_hour_feature2.csv')
other_user_hour_feature2.drop_duplicates(inplace=True)
other_user_cate_feature2 = pd.read_csv('data/other_user_cate_feature2.csv')
other_user_cate_feature2.drop_duplicates(inplace=True)
other_item_shop_feature2 = pd.read_csv('data/other_item_shop_feature2.csv')
other_item_shop_feature2.drop_duplicates(inplace=True)
other_user_brand_feature2 = pd.read_csv('data/other_user_brand_feature2.csv')
other_user_brand_feature2.drop_duplicates(inplace=True)
# other_shop_brand_feature2 = pd.read_csv('data/other_shop_brand_feature2.csv')
# other_shop_brand_feature2.drop_duplicates(inplace=True)
other_user_shop_item_feature2 = pd.read_csv('data/other_user_shop_item_feature2.csv')
other_user_shop_item_feature2.drop_duplicates(inplace=True)
other_user_item_hour_feature2 = pd.read_csv('data/other_user_item_hour_feature2.csv')
other_user_item_hour_feature2.drop_duplicates(inplace=True)
other_user_price_feature2 = pd.read_csv('data/other_user_price_feature2.csv')
other_user_price_feature2.drop_duplicates(inplace=True)
other_user_collected_feature2 = pd.read_csv('data/other_user_collected_feature2.csv')
other_user_collected_feature2.drop_duplicates(inplace=True)
other_user_sales_feature2 = pd.read_csv('data/other_user_sales_feature2.csv')
other_user_sales_feature2.drop_duplicates(inplace=True)
other_shop_hour_feature2 = pd.read_csv('data/other_shop_hour_feature2.csv')
other_shop_hour_feature2.drop_duplicates(inplace=True)

print(train2_p.shape) #(57421, 50)
train2_p = pd.merge(train2_p,user_feature2,on='user_id',how='left')
train2_p = pd.merge(train2_p,shop_feature2,on='shop_id',how='left')
train2_p = pd.merge(train2_p,item_feature2,on='item_id',how='left')
train2_p = pd.merge(train2_p,context_feature2,on='context_page_id',how='left')
train2_p = pd.merge(train2_p,user_shop_feature2,on=['user_id','shop_id'],how='left')
train2_p = pd.merge(train2_p,user_item_feature2,on=['user_id','item_id'],how='left')
train2_p = pd.merge(train2_p,user_brand_feature2,on=['user_id','item_brand_id'],how='left')
train2_p = pd.merge(train2_p,user_price_feature2,on=['user_id','item_price_level'],how='left')
train2_p = pd.merge(train2_p,user_collected_feature2,on=['user_id','item_collected_level'],how='left')
# train2_p = pd.merge(train2_p,user_pv_feature2,on=['user_id','item_pv_level'],how='left')
train2_p = pd.merge(train2_p,user_context_feature2,on=['user_id','context_page_id'],how='left')
# train2_p = pd.merge(train2_p,user_hour_feature2,on=['user_id','hour'],how='left')
train2_p = pd.merge(train2_p,user_cate_feature2,on=['user_id','cate2'],how='left')
train2_p = pd.merge(train2_p,shop_item_feature2,on=['shop_id','item_id'],how='left')
# train2_p = pd.merge(train2_p,shop_hour_feature2,on=['shop_id','hour'],how='left')
train2_p = pd.merge(train2_p,user_shop_item_feature2,on=['user_id','shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_feature2,on=['user_id'],how='left')
train2_p = pd.merge(train2_p,other_item_feature2,on=['item_id'],how='left')
train2_p = pd.merge(train2_p,other_shop_feature2,on=['shop_id'],how='left')
train2_p = pd.merge(train2_p,other_brand_feature2,on=['item_brand_id'],how='left')
train2_p = pd.merge(train2_p,other_user_item_feature2,on=['user_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_shop_feature2,on=['user_id','shop_id'],how='left')
train2_p = pd.merge(train2_p,other_user_hour_feature2,on=['user_id','hour'],how='left')
train2_p = pd.merge(train2_p,other_item_shop_feature2,on=['shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_brand_feature2,on=['user_id','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,other_shop_brand_feature2,on=['shop_id','item_brand_id'],how='left')
train2_p = pd.merge(train2_p,other_user_shop_item_feature2,on=['user_id','shop_id','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_item_hour_feature2,on=['user_id','hour','item_id'],how='left')
train2_p = pd.merge(train2_p,other_user_price_feature2,on=['user_id','item_price_level'],how='left')
train2_p = pd.merge(train2_p,other_user_collected_feature2,on=['user_id','item_collected_level'],how='left')
# train2_p = pd.merge(train2_p,other_user_sales_feature2,on=['user_id','item_sales_level'],how='left')
train2_p = pd.merge(train2_p,other_user_cate_feature2,on=['user_id','cate2'],how='left')
train2_p = pd.merge(train2_p,age_item_feature2,on=['user_age_level','item_id'],how='left')
train2_p = pd.merge(train2_p,star_item_feature2,on=['user_star_level','item_id'],how='left')
# train2_p = pd.merge(train2_p,star_shop_feature2,on=['user_star_level','shop_id'],how='left')

# train2_p = pd.merge(train2_p,occupation_item_feature2,on=['user_occupation_id','item_id'],how='left')
train2_p = pd.merge(train2_p,age_star_item_feature2,on=['user_age_level','user_star_level','item_id'],how='left')
# train2_p = pd.merge(train2_p,age_occupation_item_feature2,on=['user_age_level','user_occupation_id','item_id'],how='left')
# train2_p = pd.merge(train2_p,occupation_star_item_feature2,on=['user_occupation_id','user_star_level','item_id'],how='left')
train2_p = pd.merge(train2_p,occupation_star_age_item_feature2,on=['user_occupation_id','user_star_level','user_age_level','item_id'],how='left')
# train2_p = pd.merge(train2_p,occupation_star_age_brand_feature2,on=['user_occupation_id','user_star_level','user_age_level','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,occupation_star_age_price_feature2,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')
# train2_p = pd.merge(train2_p,age_cate_feature2,on=['user_age_level','cate2'],how='left')
train2_p = pd.merge(train2_p,star_cate_feature2,on=['user_star_level','cate2'],how='left')
# train2_p = pd.merge(train2_p,star_age_cate_feature2,on=['user_star_level','user_age_level','cate2'],how='left')
train2_p = pd.merge(train2_p,star_price_feature2,on=['user_star_level','item_price_level'],how='left')
# train2_p = pd.merge(train2_p,star_shop_feature2,on=['user_star_level','shop_id'],how='left')
# train2_p = pd.merge(train2_p,star_brand_feature2,on=['user_star_level','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,occupation_cate_feature2,on=['user_occupation_id','cate2'],how='left')
# train2_p = pd.merge(train2_p,item_brand_feature2,on=['item_id','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,occupation_brand_feature2,on=['user_occupation_id','item_brand_id'],how='left')
# train2_p = pd.merge(train2_p,star_brand_feature2,on=['item_brand_id','user_star_level'],how='left')
# train2_p = pd.merge(train2_p,gosa_feature2,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level'],how='left')
# train2_p = pd.merge(train2_p,gosa_cate_feature2,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level','cate2'],how='left')
# train2_p = pd.merge(train2_p,gosa_shop_feature2,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level','shop_id'],how='left')

print(train2_p.shape) #(57421, 176)


train3_p = pd.read_csv('data/train3_p.csv') #test文件

cate = pd.get_dummies(train3_p['cate2'])
train3_p = pd.concat([train3_p,cate],axis=1)

user_occupation_id = pd.get_dummies(train3_p['user_occupation_id'])
train3_p = pd.concat([train3_p,user_occupation_id],axis=1)


user_feature3 = pd.read_csv('data/user_feature3.csv')
user_feature3.drop_duplicates(inplace=True)
shop_feature3 = pd.read_csv('data/shop_feature3.csv')
shop_feature3.drop_duplicates(inplace=True)
item_feature3 = pd.read_csv('data/item_feature3.csv')
item_feature3.drop_duplicates(subset=['item_id'],inplace=True)#
context_feature3 = pd.read_csv('data/context_feature3.csv')
context_feature3.drop_duplicates(inplace=True)
user_shop_feature3 = pd.read_csv('data/user_shop_feature3.csv')
user_shop_feature3.drop_duplicates(inplace=True)
user_item_feature3 = pd.read_csv('data/user_item_feature3.csv')
user_item_feature3.drop_duplicates(subset=['user_id','item_id'],inplace=True)#
user_brand_feature3 = pd.read_csv('data/user_brand_feature3.csv')
user_brand_feature3.drop_duplicates(subset=['user_id','item_brand_id'],inplace=True)
user_context_feature3 = pd.read_csv('data/user_context_feature3.csv')
user_context_feature3.drop_duplicates(subset=['user_id','context_page_id'],inplace=True)
user_price_feature3 = pd.read_csv('data/user_price_feature3.csv')
user_price_feature3.drop_duplicates(subset=['user_id','item_price_level'],inplace=True)
user_collected_feature3 = pd.read_csv('data/user_collected_feature3.csv')
user_collected_feature3.drop_duplicates(subset=['user_id','item_collected_level'],inplace=True)
user_pv_feature3 = pd.read_csv('data/user_pv_feature3.csv')
user_pv_feature3.drop_duplicates(subset=['user_id','item_pv_level'],inplace=True)
user_hour_feature3 = pd.read_csv('data/user_hour_feature3.csv')
user_hour_feature3.drop_duplicates(subset=['user_id','hour'],inplace=True)
user_cate_feature3 = pd.read_csv('data/user_cate_feature3.csv')
user_cate_feature3.drop_duplicates(subset=['user_id','cate2'],inplace=True)
user_city_feature3 = pd.read_csv('data/user_city_feature3.csv')
user_city_feature3.drop_duplicates(subset=['user_id','item_city_id'],inplace=True)
shop_brand_feature3 = pd.read_csv('data/shop_brand_feature3.csv')
shop_brand_feature3.drop_duplicates(subset=['shop_id','item_brand_id'],inplace=True)
shop_item_feature3 = pd.read_csv('data/shop_item_feature3.csv')
shop_item_feature3.drop_duplicates(inplace=True)
shop_hour_feature3 = pd.read_csv('data/shop_hour_feature3.csv')
shop_hour_feature3.drop_duplicates(inplace=True)
user_shop_item_feature3 = pd.read_csv('data/user_shop_item_feature3.csv')
user_shop_item_feature3.drop_duplicates(inplace=True)
age_item_feature3 = pd.read_csv('data/age_item_feature3.csv')
age_item_feature3.drop_duplicates(subset=['user_age_level','item_id'],inplace=True)
star_item_feature3 = pd.read_csv('data/star_item_feature3.csv')
star_item_feature3.drop_duplicates(subset=['user_star_level','item_id'],inplace=True)
occupation_item_feature3 = pd.read_csv('data/occupation_item_feature3.csv')
occupation_item_feature3.drop_duplicates(subset=['user_occupation_id','item_id'],inplace=True)
age_occupation_item_feature3 = pd.read_csv('data/age_occupation_item_feature3.csv')
age_occupation_item_feature3.drop_duplicates(subset=['user_age_level','user_occupation_id','item_id'],inplace=True)
age_star_item_feature3 = pd.read_csv('data/age_star_item_feature3.csv')
age_star_item_feature3.drop_duplicates(subset=['user_age_level','user_star_level','item_id'],inplace=True)
occupation_star_item_feature3 = pd.read_csv('data/occupation_star_item_feature3.csv')
occupation_star_item_feature3.drop_duplicates(subset=['user_occupation_id','user_star_level','item_id'],inplace=True)
occupation_star_age_item_feature3 = pd.read_csv('data/occupation_star_age_item_feature3.csv')
occupation_star_age_item_feature3.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_id'],inplace=True)
age_cate_feature3 = pd.read_csv('data/age_cate_feature3.csv')
age_cate_feature3.drop_duplicates(subset=['user_age_level','cate2'],inplace=True)
star_cate_feature3 = pd.read_csv('data/star_cate_feature3.csv')
star_cate_feature3.drop_duplicates(subset=['user_star_level','cate2'],inplace=True)
occupation_cate_feature3 = pd.read_csv('data/occupation_cate_feature3.csv')
occupation_cate_feature3.drop_duplicates(subset=['user_occupation_id','cate2'],inplace=True)
item_brand_feature3 = pd.read_csv('data/item_brand_feature3.csv')
item_brand_feature3.drop_duplicates(subset=['item_id','item_brand_id'],inplace=True)
occupation_brand_feature3 = pd.read_csv('data/occupation_brand_feature3.csv')
occupation_brand_feature3.drop_duplicates(subset=['user_occupation_id','item_brand_id'],inplace=True)
star_brand_feature3 = pd.read_csv('data/star_brand_feature3.csv')
star_brand_feature3.drop_duplicates(subset=['user_star_level','item_brand_id'],inplace=True)
occupation_star_age_brand_feature3 = pd.read_csv('data/occupation_star_age_brand_feature3.csv')
occupation_star_age_brand_feature3.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_brand_id'],inplace=True)
occupation_star_age_price_feature3 = pd.read_csv('data/occupation_star_age_price_feature3.csv')
occupation_star_age_price_feature3.drop_duplicates(subset=['user_occupation_id','user_star_level','user_age_level','item_price_level'],inplace=True)
star_price_feature3 = pd.read_csv('data/star_price_feature3.csv')
star_price_feature3.drop_duplicates(subset=['user_star_level','item_price_level'],inplace=True)
star_shop_feature3 = pd.read_csv('data/star_shop_feature3.csv')
star_shop_feature3.drop_duplicates(subset=['user_star_level','shop_id'],inplace=True)
gosa_feature3 = pd.read_csv('data/gosa_feature3.csv')
gosa_feature3.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level'],inplace=True)
gosa_shop_feature3 = pd.read_csv('data/gosa_shop_feature3.csv')
gosa_shop_feature3.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level','shop_id'],inplace=True)
gosa_cate_feature3 = pd.read_csv('data/gosa_cate_feature3.csv')
gosa_cate_feature3.drop_duplicates(subset=['user_gender_id','user_occupation_id','user_star_level','user_age_level','cate2'],inplace=True)
star_age_cate_feature3 = pd.read_csv('data/star_age_cate_feature3.csv')
star_age_cate_feature3.drop_duplicates(subset=['user_star_level','user_age_level','cate2'],inplace=True)

other_user_feature3 = pd.read_csv('data/other_user_feature3.csv')
other_user_feature3.drop_duplicates(inplace=True)
other_item_feature3 = pd.read_csv('data/other_item_feature3.csv')
other_item_feature3.drop_duplicates(inplace=True)
other_shop_feature3 = pd.read_csv('data/other_shop_feature3.csv')
other_shop_feature3.drop_duplicates(inplace=True)
other_brand_feature3 = pd.read_csv('data/other_brand_feature3.csv')
other_brand_feature3.drop_duplicates(inplace=True)
other_user_item_feature3 = pd.read_csv('data/other_user_item_feature3.csv')
other_user_item_feature3.drop_duplicates(inplace=True)
other_user_shop_feature3 = pd.read_csv('data/other_user_shop_feature3.csv')
other_user_shop_feature3.drop_duplicates(inplace=True)
other_user_hour_feature3 = pd.read_csv('data/other_user_hour_feature3.csv')
other_user_hour_feature3.drop_duplicates(inplace=True)
other_user_cate_feature3 = pd.read_csv('data/other_user_cate_feature3.csv')
other_user_cate_feature3.drop_duplicates(inplace=True)
other_item_shop_feature3 = pd.read_csv('data/other_item_shop_feature3.csv')
other_item_shop_feature3.drop_duplicates(inplace=True)
other_user_brand_feature3 = pd.read_csv('data/other_user_brand_feature3.csv')
other_user_brand_feature3.drop_duplicates(inplace=True)
# other_shop_brand_feature3 = pd.read_csv('data/other_shop_brand_feature3.csv')
# other_shop_brand_feature3.drop_duplicates(inplace=True)
other_user_shop_item_feature3 = pd.read_csv('data/other_user_shop_item_feature3.csv')
other_user_shop_item_feature3.drop_duplicates(inplace=True)
other_user_item_hour_feature3 = pd.read_csv('data/other_user_item_hour_feature3.csv')
other_user_item_hour_feature3.drop_duplicates(inplace=True)
other_user_price_feature3 = pd.read_csv('data/other_user_price_feature3.csv')
other_user_price_feature3.drop_duplicates(inplace=True)
other_user_collected_feature3 = pd.read_csv('data/other_user_collected_feature3.csv')
other_user_collected_feature3.drop_duplicates(inplace=True)
other_user_sales_feature3 = pd.read_csv('data/other_user_sales_feature3.csv')
other_user_sales_feature3.drop_duplicates(inplace=True)
other_shop_hour_feature3 = pd.read_csv('data/other_shop_hour_feature3.csv')
other_shop_hour_feature3.drop_duplicates(inplace=True)
print(train3_p.shape) #(18371, 49)
train3_p = pd.merge(train3_p,user_feature3,on='user_id',how='left')
train3_p = pd.merge(train3_p,shop_feature3,on='shop_id',how='left')
train3_p = pd.merge(train3_p,item_feature3,on='item_id',how='left')
train3_p = pd.merge(train3_p,context_feature3,on='context_page_id',how='left')
train3_p = pd.merge(train3_p,user_shop_feature3,on=['user_id','shop_id'],how='left')
train3_p = pd.merge(train3_p,user_item_feature3,on=['user_id','item_id'],how='left')
train3_p = pd.merge(train3_p,user_brand_feature3,on=['user_id','item_brand_id'],how='left')
train3_p = pd.merge(train3_p,user_price_feature3,on=['user_id','item_price_level'],how='left')
train3_p = pd.merge(train3_p,user_collected_feature3,on=['user_id','item_collected_level'],how='left')
# train1_p = pd.merge(train1_p,user_pv_feature1,on=['user_id','item_pv_level'],how='left')
train3_p = pd.merge(train3_p,user_context_feature3,on=['user_id','context_page_id'],how='left')
# train1_p = pd.merge(train1_p,user_hour_feature1,on=['user_id','hour'],how='left')
train3_p = pd.merge(train3_p,user_cate_feature3,on=['user_id','cate2'],how='left')
train3_p = pd.merge(train3_p,shop_item_feature3,on=['shop_id','item_id'],how='left')
# train1_p = pd.merge(train1_p,shop_hour_feature1,on=['shop_id','hour'],how='left')
train3_p = pd.merge(train3_p,user_shop_item_feature3,on=['user_id','shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_feature3,on=['user_id'],how='left')
train3_p = pd.merge(train3_p,other_item_feature3,on=['item_id'],how='left')
train3_p = pd.merge(train3_p,other_shop_feature3,on=['shop_id'],how='left')
train3_p = pd.merge(train3_p,other_brand_feature3,on=['item_brand_id'],how='left')
train3_p = pd.merge(train3_p,other_user_item_feature3,on=['user_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_shop_feature3,on=['user_id','shop_id'],how='left')
train3_p = pd.merge(train3_p,other_user_hour_feature3,on=['user_id','hour'],how='left')
train3_p = pd.merge(train3_p,other_item_shop_feature3,on=['shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_brand_feature3,on=['user_id','item_brand_id'],how='left')
# train1_p = pd.merge(train1_p,other_shop_brand_feature1,on=['shop_id','item_brand_id'],how='left')
train3_p = pd.merge(train3_p,other_user_shop_item_feature3,on=['user_id','shop_id','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_item_hour_feature3,on=['user_id','hour','item_id'],how='left')
train3_p = pd.merge(train3_p,other_user_price_feature3,on=['user_id','item_price_level'],how='left')
train3_p = pd.merge(train3_p,other_user_collected_feature3,on=['user_id','item_collected_level'],how='left')
# train1_p = pd.merge(train1_p,other_user_sales_feature1,on=['user_id','item_sales_level'],how='left')
train3_p = pd.merge(train3_p,other_user_cate_feature3,on=['user_id','cate2'],how='left')
train3_p = pd.merge(train3_p,age_item_feature1,on=['user_age_level','item_id'],how='left')
train3_p = pd.merge(train3_p,star_item_feature3,on=['user_star_level','item_id'],how='left')
# train3_p = pd.merge(train3_p,star_shop_feature3,on=['user_star_level','shop_id'],how='left')
# train3_p = pd.merge(train3_p,occupation_item_feature3,on=['user_occupation_id','item_id'],how='left')
train3_p = pd.merge(train3_p,age_star_item_feature3,on=['user_age_level','user_star_level','item_id'],how='left')
# train1_p = pd.merge(train1_p,age_occupation_item_feature1,on=['user_age_level','user_occupation_id','item_id'],how='left')
# train3_p = pd.merge(train3_p,occupation_star_item_feature3,on=['user_occupation_id','user_star_level','item_id'],how='left')
train3_p = pd.merge(train3_p,occupation_star_age_item_feature3,on=['user_occupation_id','user_star_level','user_age_level','item_id'],how='left')
# train3_p = pd.merge(train3_p,occupation_star_age_brand_feature3,on=['user_occupation_id','user_star_level','user_age_level','item_brand_id'],how='left')
# train3_p = pd.merge(train3_p,occupation_star_age_price_feature3,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')

# train3_p = pd.merge(train3_p,occupation_star_age_price_feature3,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')
# train3_p = pd.merge(train3_p,occupation_star_age_cate_feature3,on=['user_occupation_id','user_star_level','user_age_level','cate2'],how='left')

# train3_p = pd.merge(train3_p,age_cate_feature3,on=['user_age_level','cate2'],how='left')
train3_p = pd.merge(train3_p,star_cate_feature3,on=['user_star_level','cate2'],how='left')
# train3_p = pd.merge(train3_p,star_age_cate_feature3,on=['user_star_level','user_age_level','cate2'],how='left')
train3_p = pd.merge(train3_p,star_price_feature3,on=['user_star_level','item_price_level'],how='left')
# train3_p = pd.merge(train3_p,star_shop_feature3,on=['user_star_level','shop_id'],how='left')
# train3_p = pd.merge(train3_p,star_brand_feature3,on=['user_star_level','item_brand_id'],how='left')
# train3_p = pd.merge(train3_p,occupation_cate_feature3,on=['user_occupation_id','cate2'],how='left')
# train3_p = pd.merge(train3_p,item_brand_feature3,on=['item_id','item_brand_id'],how='left')
# train3_p = pd.merge(train3_p,star_brand_feature3,on=['item_brand_id','user_star_level'],how='left')
# train3_p = pd.merge(train3_p,gosa_feature3,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level'],how='left')
# train3_p = pd.merge(train3_p,gosa_cate_feature3,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level','cate2'],how='left')

# train3_p = pd.merge(train3_p,gosa_shop_feature3,on=['user_gender_id','user_occupation_id','user_star_level','user_age_level','shop_id'],how='left')

print(train3_p.shape) #(18371, 174)

train3_pre = train3_p[['instance_id']]
# train2_pre = train2_p[['instance_id']]
'''

drop_ele = [
            'instance_id','item_id','item_property_list','item_brand_id','item_pv_level','item_city_id',
            'user_id','user_occupation_id',
           'context_id','context_timestamp','predict_category_property','shop_id','cate2','cate3',
           'user_click_min','user_click_mean'
          ,'user_am_click','collected_user_tcrate',
           'user_brand_today_click','is_high_sale','user_price_crate'
           #    'user_age_level', 'user_star_level', 'shop_review_num_level',
           # 'shop_star_level',
            ]
#  '2.27313E+16', '5.0966E+17', '1.96806E+18', '2.01198E+18', '2.43672E+18','user_occupation_id',
#            '3.20367E+18', '4.87972E+18', '5.75569E+18', '5.79935E+18', '-1', '2004', '2005','ubefore',

# ,'user_cate_buy', 'max_sale_hour', 'min_age', 'sale_am',, 'user_shop_click_buy_rate',
#              'user_shop_click_buy_rate.1',\
#             'u_i_s_tclick_rate',
#            'sales_user_tcrate', 'item_user_rate', 'user_shop_today', 'shop_user_rate',
#          , 'user_shop_itnum','user_collected_ctotal','user_price_ctotal',
#
#              'user_cate_click',
train1_p_y = train1_p.is_trade
train1_p_x = train1_p.drop(drop_ele,axis=1)
train1_p_x.corr().to_csv('cor.csv')
'''
train2_p_y = train2_p.is_trade
train2_p_x =train2_p.drop(drop_ele,axis=1)

train12 = pd.concat([train1_p_x,train2_p_x],axis=0)
train12_y = train12.is_trade
train3_p = train3_p.drop(drop_ele,axis=1)

train2_p_x = train2_p_x.drop('is_trade',axis=1)
train1_p_x = train1_p_x.drop('is_trade',axis=1)
# train12 = train12.drop(['is_trade','min_age'],axis=1)
train12 = train12.drop(['is_trade'],axis=1)
train12_gdbt = train12
'''
# print(train12.columns.values.tolist()) #获得列名列表
# print(train3_p.columns.values.tolist())
# exit(0)

train1_p = xgb.DMatrix(train1_p_x, label=train1_p_y)
train2_p = xgb.DMatrix(train2_p_x, label=train2_p_y)
train12 = xgb.DMatrix(train12 , label=train12_y)
train3_p = xgb.DMatrix(train3_p)


params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'scale_pos_weight': 1,
          'eval_metric': 'logloss', #对于有效数据的度量方法。
                # 典型值有：rmse 均方根误差；mae 平均绝对误差；logloss 负对数似然函数值；error 二分类错误率 (阈值为 0.5)；merror 多分类错误率；mlogloss 多分类
          'gamma': 0.1, #原来是0.1
          'min_child_weight': 1.0,
          'max_depth': 4, #应该为4
          'lambda': 15,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.008, #learning_rate
          'tree_method': 'exact',
          'seed': 1,
          'nthread': 6
          }
watchlist = [(train1_p,'train'),(train2_p,'val')]
model = xgb.train(params,train1_p,num_boost_round=2000,evals=watchlist,early_stopping_rounds=100)
pre1 = model.predict(train2_p)
# processHasTrade(train2_p, pre1)
train2_pre = pd.DataFrame(index=None)
print(log_loss(train2_p_y,pre1))

# train1_p = lgb.Dataset(np.array(train1_p_x), label=train1_p_y)#转为lightgbm格式
# train2_p =np.column_stack((train2_p_x,train2_p_y))
# train12 = lgb.Dataset(np.array(train12) , label=train12_y)
# train3_p =np.array(train3_p)
# params = {'boosting_type': 'gbdt',
#           'num_leaves': 3,
#           'max_depth': 150,
#           'n_estimators': 100,
#           'max_bin': 55,
#           'n_jobs': 20,
#           'bagging_fraction': 0.8,
#           'bagging_freq': 5,
#           'feature_fraction': 0.2319,
#           'feature_fraction_seed': 9,
#           'bagging_seed': 9,
#           'min_data_in_leaf': '6',
#           'min_sum_hessian_in_leaf': 11,
#           'nthread': 6
#           }
# model_lgb=lgb.train(params,train1_p,num_boost_round=2000)
# pre1 = model_lgb.predict(train2_p)
# train2_pre = pd.DataFrame(index=None)
# print(log_loss(train2_p_y,pre1))

# def my(a):
#     if a<0.5:
#         if a-0.01>=0:
#             a=a-0.01
#         else:
#             a=0
# train2_pre['pre'] = pre1
# train2_pre['pre'].apply(my)
# pre1 = train2_pre['pre']
# print(log_loss(train2_p_y,pre1))
# train2_pre['pre'] = pre1
# train2_pre.to_csv('train2_pre_xgb.csv',index=None)

watchlist = [(train12,'train')]
model = xgb.train(params,train12,num_boost_round=1700,evals=watchlist)
train3_pre['predicted_score'] = model.predict(train3_p)
# processHasTrade(train3_p, "predicted_score")
train3_pre.to_csv('xgb_adv_pred19.csv',sep=' ',index=None)



# train1_p_x.fillna(0,inplace=True)
# train2_p_x.fillna(0,inplace=True)
# train12.fillna(0,inplace=True)
# train3_p.fillna(0,inplace=True)
# clf1 = GradientBoostingClassifier(
#     loss='deviance',  ##损失函数默认deviance  deviance具有概率输出的分类的偏差
#     n_estimators=230, ##默认100 回归树个数 弱学习器个数
#     learning_rate=0.008,  ##默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长
#     max_depth=6,   ## 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
#     subsample=0.8,  ##树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
#     # min_impurity_decrease=1e-7, ##停止分裂叶子节点的阈值
#     verbose=2,  ##打印输出 大于1打印每棵树的进度和性能
#     warm_start=False, ##True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
#     random_state=0,  ##随机种子-方便重现
# )
# clf1.fit(train1_p_x,train1_p_y)
# pre = clf1.predict_proba(train2_p_x)
# clf1.fit(train12,train12_y)
# pre = clf1.predict_proba(train3_p)
# gbdt_adv_pre = pd.DataFrame(index=None)
# gbdt_adv_pre['instance_id'] = train3_pre['instance_id']
# gbdt_adv_pre['predicted_score'] = pre[:,1]
# gbdt_adv_pre.to_csv('gbdt_adv_pre.csv')

#  xgb1 = xgb.XGBClassifier(
#  learning_rate = 0.01,
#  n_estimators= 400,
#  max_depth= 5,
#  min_child_weight= 2,
#  #gamma=1,
#  gamma=0.9,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread= -1,
#  scale_pos_weight=1,
# )
# lr = LogisticRegression()
# xgb = xgb.XGBClassifier(
#           # booster = 'gbtree',
#           objective = 'binary:logistic',
#           scale_pos_weight = 1,
#           gamma = 0.1,
#           min_child_weight = 1.1,
#           max_depth = 5,
#           # lambda = 15,
#           reg_lambda = 15,
#           n_estimators = 800,
#           subsample = 0.8,
#           colsample_bytree = 0.7,
#           colsample_bylevel = 0.7,
#           learning_rate = 0.01,
#           seed = 0,
#           nthread = 4,
#           silent = 0
# )
#
#
# sclf = StackingClassifier(classifiers=[xgb,clf1],
#                           use_probas=True,
#                           average_probas=False,
#                           verbose = 1,
#                           meta_classifier=xgb1)
# sclf.fit(train1_p_x,train1_p_y)
# pre = sclf.predict_proba(train1_p_x)
# print(log_loss(train1_p_y,pre))
# pre = sclf.predict_proba(train2_p_x)
# print(log_loss(train2_p_y,pre))
#
# # watchlist = [(train12,'train')]
# # model = xgb.train(params,train12,num_boost_round=1700,evals=watchlist)
# # train3_pre['predicted_score'] = model.predict(train3_p)
# # # processHasTrade(train3_p, "predicted_score")
# # train3_pre.to_csv('xgb_adv_pred17.csv',sep=' ',index=None)
# sclf.fit(train12,train12_y)
#
# train3_pre['predicted_score']  = sclf.predict_proba(train3_p)
# train3_pre.to_csv('stacking-ctr18.csv',sep=' ',index=None)





    # param_test = {
#     'n_estimators': range(2000, 3000, 100),
#     'max_depth': range(4, 7, 1)
# }
# model =clf
#
# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(estimator = model, param_grid = param_test, scoring='accuracy', cv=5)
# grid_search.fit(train1_p_x, train1_p_y)
# print(grid_search.best_params_, grid_search.best_score_)



#GBDT+LR:
# 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
# X_train, X_train_lr, y_train, y_train_lr = train_test_split(train1_p_x, train1_p_y, test_size=0.5)
# imp = Imputer(missing_values='NaN',strategy='mean',axis=0,verbose=0,copy=True) #missing_values：缺失值，可以为整数或 NaN(缺失值 numpy.nan 用字符串‘NaN’表示)，默认为 NaN
# X_train = imp.fit_transform(X_train)
# X_train_lr = imp.fit_transform(X_train_lr)
# train2_p_x = imp.fit_transform(train2_p_x)

#调参1：n_estimators=40时最优
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.01, min_samples_split=300,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
#                        param_grid = {'n_estimators':[10,20,30,40,50,60,70,80]}, scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(X_train,y_train)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
#调参2：'max_depth'=7 和 'min_samples_split'=300
# gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.01, n_estimators=40, min_samples_leaf=20,
#       max_features='sqrt', subsample=0.8, random_state=10),
#    param_grid = {'max_depth':[3,4,5,6,7,8,9,10], 'min_samples_split':[100,200,300,400,500,600]}, scoring='roc_auc',iid=False, cv=5)
# gsearch2.fit(X_train,y_train)
# print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
#调参3：'min_samples_leaf': 40, 'min_samples_split': 600
# gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=40,max_depth=7,
#                                      max_features='sqrt', subsample=0.8, random_state=10),
#                        param_grid ={'min_samples_split':[100,200,300,400,500,600,700,800], 'min_samples_leaf':[10,20,30,40,50,60]}, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X_train,y_train)
# print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
#调参4：'subsample': 0.6
# gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.01, n_estimators=60,max_depth=7, min_samples_leaf =40,
#                 min_samples_split =600,  random_state=10),
#                 param_grid = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}, scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X_train,y_train)
# print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

#grd = GradientBoostingClassifier(n_estimators=10, learning_rate=0.008)


# grd = GradientBoostingClassifier(n_estimators=10, learning_rate=0.01,
#                                  max_depth=7, max_features='sqrt',subsample = 0.6,
#                                  min_samples_leaf=40, min_samples_split=300, random_state=5)
# # 调用one-hot编码
# grd_enc = preprocessing.OneHotEncoder()
# # 调用LR分类模型
# grd_lm = LogisticRegression()
# #使用X_train训练GBDT模型，后面用此模型构造特征
# grd.fit(X_train, y_train)
#
# # fit one-hot编码器
# grd_enc.fit(grd.apply(X_train)[:, :, 0])
#
# #使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练
# grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# # 用训练好的LR模型多X_test做预测
# y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(train2_p_x)[:, :, 0]))[:, 1]
# # 根据预测结果输出
# print(log_loss(train2_p_y,y_pred_grd_lm))




#lightgbm+xgboost
# train1_p = xgb.DMatrix(train1_p_x, label=train1_p_y)
# train2_p = xgb.DMatrix(train2_p_x, label=train2_p_y)
# train12 = xgb.DMatrix(train12 , label=train12_y)
# train3_p = xgb.DMatrix(train3_p)
# clf = lgb.LGBMClassifier(boosting_type='gbdt', #num_leaves=400,
#                          max_depth=8, learning_rate=0.008, n_estimators=1000,
#                          n_jobs=4)
# clf.fit(train1_p_x, train1_p_y, eval_set=[(train2_p_x, train2_p_y)],
#         eval_metric='binary_logloss',
#         #categorical_feature=['user_gender_id', ],
#         early_stopping_rounds=150)
# lgb_pred = clf.predict_proba(train2_p_x, num_iteration=clf.best_iteration_, )[:, 1]
#train2_p_y['lgb_predict'] = lgb_pred
# print(len(train[features]))
# print(train[features])

# xgb_train = xgb.DMatrix(train1_p_x, label=train1_p_y)
# xgb_test = xgb.DMatrix(train2_p_x, label=train2_p_y)
# # print(xgb_train)
# param = {'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 'silent': 1,
#          'seed': 20, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'nthread': 4}
# model = xgb.train(params=param, dtrain=xgb_train, num_boost_round=2000, evals=[(xgb_test, 'val')],
#                   early_stopping_rounds=100, verbose_eval=True)
# # print(model.best_iteration)
# # make prediction
# xgb_pred = model.predict(xgb_test, ntree_limit=model.best_iteration)
# train1_p = xgb.DMatrix(train1_p_x, label=train1_p_y)
# train2_p = xgb.DMatrix(train2_p_x, label=train2_p_y)
# train12 = xgb.DMatrix(train12 , label=train12_y)
# train3_p = xgb.DMatrix(train3_p)
#
#
# params = {'booster': 'gbtree',
#           'objective': 'binary:logistic',
#           'scale_pos_weight': 1,
#           'eval_metric': 'logloss', #对于有效数据的度量方法。
#                 # 典型值有：rmse 均方根误差；mae 平均绝对误差；logloss 负对数似然函数值；error 二分类错误率 (阈值为 0.5)；merror 多分类错误率；mlogloss 多分类
#           'gamma': 0.1, #原来是0.1
#           'min_child_weight': 1.0,
#           'max_depth': 4, #应该为4
#           'lambda': 15,
#           'subsample': 0.7,
#           'colsample_bytree': 0.7,
#           'colsample_bylevel': 0.7,
#           'eta': 0.008, #learning_rate
#           'tree_method': 'exact',
#           'seed': 1,
#           'nthread': 6
#           }
# watchlist = [(train1_p,'train'),(train2_p,'val')]
# model = xgb.train(params,train1_p,num_boost_round=2000,evals=watchlist,early_stopping_rounds=100)
# xgb_pre1 = model.predict(train2_p)
# train2_pre = pd.DataFrame(index=None)
# # print(lgb_pred)
# # print(xgb_pre1)
# print(log_loss(train2_p_y, lgb_pred))
# print(log_loss(train2_p_y, xgb_pre1))
# print(log_loss(train2_p_y, (lgb_pred + xgb_pre1) / 2))


