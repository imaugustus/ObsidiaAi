import keras
from keras import backend as K
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.models import Model
import pickle
import re
import numpy as np
import pandas as pd
import math


# Loading stock data from pickle
intern = pickle.load(open(r'D:/Data/intern.pkl', 'rb'))
MktData = intern['MktData']
InstrumentInfo = intern['InstrumentInfo']
MktData = MktData.swaplevel(0, 1, axis=1)


# 中位数去极值
def filter_extreme(factor_section, n=5):
    Dm = factor_section.median()
    Dm1 = ((factor_section-Dm).abs()).median()
    max_limit = Dm + n*Dm1
    min_limit = Dm - n*Dm1
    factor_section = np.clip(factor_section, min_limit, max_limit)
    return factor_section


# 标准化
def normalize(factor_section):
    mean = factor_section.mean()
    std = factor_section.std()
    factor_section = (factor_section - mean)/std
    return factor_section


# 缺失值处理
def fill_na(factor_section):
    factor_section = factor_section.fillna(factor_section.mean())
    return factor_section


# 预处理因子
def preprocess_factor(factor):
    preprocessed_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
    for date in factor.index:
        section_factor = factor.loc[date, :]
        section_factor = filter_extreme(section_factor)
        section_factor = normalize(section_factor)
        section_factor = fill_na(section_factor)
        preprocessed_factor.loc[date, :] = section_factor
    return preprocessed_factor


# 获取行业后代
def get_industry_descendant(industry_code, descendant_order=0):
    pattern = re.compile(industry_code[0:3+descendant_order])
    descendant = []
    for stock_code, representation in zip(InstrumentInfo.index, InstrumentInfo['SWICS']):
        if re.match(pattern, representation):
            descendant.append(stock_code)
    return descendant


# 获取行业delta因子
def data_input(industry_code='430000', start_date='2016-01-04', time_step=10, train_day=20, ratio=0.8):
    descendant = get_industry_descendant(industry_code, descendant_order=0)
    ret = MktData.loc[:, (descendant, 'ret')]
    ret.columns = ret.columns.droplevel(level=1)
    industry_mean = ret.mean(axis=1)
    relative_ret = ret.subtract(industry_mean, axis=0)
    factor = preprocess_factor(relative_ret)
    start_index = factor.index.get_loc(start_date)
    end_index = start_index + train_day
    X_set = []
    y_set = []
    for i in range(train_day):
        now_start_index = start_index + i
        now_end_index = now_start_index + time_step
        X = factor.iloc[now_start_index:now_end_index, :].as_matrix()
        y = factor.iloc[now_end_index, :].as_matrix().reshape(1, 111)
        X_set.append(X)
        y_set.append(y)
    np.random.seed(12)
    permuted_index = np.random.permutation(range(len(X_set)))
    permuted_index = [np.asscalar(item) for item in permuted_index]
    train_set_number = round(len(X_set)*ratio)
    train_index = permuted_index[:train_set_number]
    test_index = permuted_index[train_set_number:]
    X_train = [X_set[i] for i in train_index]
    X_test = [X_set[i] for i in test_index]
    y_train = [y_set[i] for i in train_index]
    y_test = [y_set[i] for i in test_index]
    return X_train, y_train, X_test, y_test

def model(time_step=10, learning_rate=0.01, iterations=400):
    factor_ts = Input(shape=time_step, dtype='float')
    x = LSTM(units=128)(factor_ts)
    x = Dropout(0.5)(x)
    x = LSTM(units=128)(x)
    x = Dropout(0.5)(x)
    x = Dense(units=1)(x)
    x = Activation("linear")(x)
    model = Model(inputs=factor_ts, outputs=x)
    return model





if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_input('720000', time_step=10, train_day=10, ratio=0.7)


# TODO 因子作为输入层是可以确定的， 输出层采用输入层后一天的收益率？ many-to-one 结构
# TODO 历史因子采用10天：time-step=10 即10-1结构
# TODO 中间采用双层隐藏层---参考华泰金工人工智能系列，并采用单向神经网络
# TODO 训练集大小为200，滚动向后取训练集
# TODO 但是该如何处理多支股票的问题呢--每天的预测权重都在变化--每天股票池中的所有stock作为一次训练的样本

