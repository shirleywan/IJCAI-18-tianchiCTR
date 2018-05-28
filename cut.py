import pandas as pd
import xgboost as xgb
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import time
from datetime import date
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# train = pd.read_table('data/round2_train.txt',delim_whitespace=True)
# test = pd.read_table('data/round2_test.txt',delim_whitespace=True)
train = pd.read_table('newdata10/train2ac.txt',delim_whitespace=True)
train.user_gender_id = train.user_gender_id.replace(-1,0)#缺失值用0代替

train.item_brand_id = train.item_brand_id.replace(-1,train['item_brand_id'].mode())

train.item_city_id = train.item_city_id.replace(-1,train['item_city_id'].mode())

train.loc[train.item_sales_level==-1,'item_sales_level']=pd.DataFrame(train[train['item_sales_level'].isin([-1])]['item_collected_level'],columns=['item_collected_level']).reset_index()['item_collected_level']

train.user_gender_id = train.user_gender_id.replace(-1,train['user_gender_id'].mode())

train.user_age_level = train.user_age_level.replace(-1,train['user_age_level'].mode())

train.user_occupation_id = train.user_occupation_id.replace(-1,train['user_occupation_id'].mode())

train.user_star_level = train.user_star_level.replace(-1,train['user_star_level'].mode())

train.shop_review_positive_rate = train.shop_review_positive_rate.replace(-1,1)

train.shop_score_service = train.shop_score_service.replace(-1,1)

train.shop_score_delivery = train.shop_score_delivery.replace(-1,1)

train.shop_score_description = train.shop_score_description.replace(-1,1)

# 首先将商品类目切分
def split_item_category(s):
    cate = s.split(';')
    if len(cate)==3:
        return cate
    elif len(cate)<3:
        cate.append('-1')

    return cate
train['cate']=train['item_category_list'].apply(split_item_category)
train['cate2'] = train['cate'].apply(lambda x:x[1])#取出第一列
train['cate3'] = train['cate'].apply(lambda x:x[2])#取出第二列
train['cate2'].replace('-1',method='ffill') #向前填充
train['cate3'].replace('-1',method='ffill')
train.drop(['cate'],axis=1,inplace=True)
#test.drop(['cate','item_category_list'],axis=1,inplace=True)

# 将时间戳转化为时间格式
def timestamp_datetime(value):
    format = '%Y%m%d%H%M%S'
    value = time.localtime(value)
    ## 经过localtime转换后变成
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最后再经过strftime函数转换为正常日期格式。
    dt = time.strftime(format, value)
    return dt
# 将unix时间戳转为正常格式
train['context_timestamp'] = train['context_timestamp'].apply(timestamp_datetime)

train['is_weekend'] = train['context_timestamp'].apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1).apply(lambda x:1 if x in (6,7) else 0)

train['day']=train['context_timestamp'].apply(lambda x:int(x[6:8]))#day，取出日期
train['hour']=train['context_timestamp'].apply(lambda x:int(x[8:10])) #hour
train['is_am']=train['hour'].apply(lambda x:1 if x<12 else 0) #上午

# train.drop('context_timestamp',axis=1,inplace=True)
# test.drop('context_timestamp',axis=1,inplace=True)


# 将double切换为float
train[['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']]=train[['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']].astype(float)#double转为float


# 首先将商品类目切分
def split_property(s):
    cate = s.split(';')
#    print(len(cate))
    if len(cate) == 20:
        return cate
    elif len(cate) < 20:
        for i in range(20-len(cate)):
            cate.append('-1')
    return cate
# 将商品属性切分
train['prop']=train['item_property_list'].apply(split_property)
train['prop1'] = train['prop'].apply(lambda x:x[0])
train['prop2'] = train['prop'].apply(lambda x:x[1])
train['prop3'] = train['prop'].apply(lambda x:x[2])
train['prop4'] = train['prop'].apply(lambda x:x[3])
train['prop5'] = train['prop'].apply(lambda x:x[4])
train['prop6'] = train['prop'].apply(lambda x:x[5])
train['prop7'] = train['prop'].apply(lambda x:x[6])
train['prop8'] = train['prop'].apply(lambda x:x[7])
train['prop9'] = train['prop'].apply(lambda x:x[8])
train['prop10'] = train['prop'].apply(lambda x:x[9])
train['prop11'] = train['prop'].apply(lambda x:x[10])
train['prop12'] = train['prop'].apply(lambda x:x[11])
train['prop13'] = train['prop'].apply(lambda x:x[12])
train['prop14'] = train['prop'].apply(lambda x:x[13])
train['prop15'] = train['prop'].apply(lambda x:x[14])
train['prop16'] = train['prop'].apply(lambda x:x[15])
train['prop17'] = train['prop'].apply(lambda x:x[16])
train['prop18'] = train['prop'].apply(lambda x:x[17])
train['prop19'] = train['prop'].apply(lambda x:x[18])
train['prop20'] = train['prop'].apply(lambda x:x[19])

# train.drop(['prop','item_property_list'],axis=1,inplace=True)
# test.drop(['prop','item_property_list'],axis=1,inplace=True)
train.drop(['prop'],axis=1,inplace=True)


train.to_csv('newdata/3train_cut.csv', index=None)
