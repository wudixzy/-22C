import numpy as np
import pandas as pd
import re

def fix_date_format(date_str):
    # 如果日期以 '200' 开头，则删除 '200'
    if isinstance(date_str, str) and re.match(r'^200\d{1}/\d{1,2}/\d{1,2}$', date_str):
        return date_str[3:]
    # 如果日期以 '20' 开头，则删除 '20'
    elif isinstance(date_str, str) and re.match(r'^20\d{2}/\d{1,2}/\d{1,2}$', date_str):
        return date_str[2:]
    else:
        # 如果不是上述两种情况，则保持原样
        return date_str

class Portfolio():
    def constructor(self):
        self.assets = {"bitcoin":0, "gold":0}
        self.cash = 1000
    


class Market():
    def __init__(self):
        self.data_BCHAIN = pd.read_csv("processed_BCHAIN-MKPRU.csv")
        self.data_GOLD = pd.read_csv("processed_data_gold.csv")
        self.data_BCHAIN = pd.DataFrame(self.data_BCHAIN)
        self.data_GOLD = pd.DataFrame(self.data_GOLD)

        self.weight_factor = 0.1
        self.C = 1 # 计算权重因子的常数
        self.t = 10 # 计算权重因子的范围

        self.n_moving_average_BCHAIN = 10
        self.n_moving_average_GOLD = 10 # 与权重因子的范围保持一致

        self.thr_BCHAIN = 0.05
        self.thr_GOLD = 0.05

        self.rate_above_moving_average_GOLD = 1.05
        self.rate_above_moving_average_BCHAIN = 1.05

        self.rate_under_moving_average_GOLD = 0.90
        self.rate_under_moving_average_BCHAIN = 0.90

        self.rate_Extreme_Market = 5
        self.size_window_Extreme_Market = 10

    def grad(self):
        self.data_BCHAIN['grad'] = 0
        self.data_GOLD['grad'] = 0

        self.data_BCHAIN['grad'] = self.data_BCHAIN['value'] - self.data_BCHAIN['value'].shift(1)
        self.data_GOLD['grad'] = self.data_GOLD['value'] - self.data_GOLD['value'].shift(1)
    
    def weight_factor(self):
        days = list(range(self.t+1)) + 1
        self.weight_factor = self.C / days**2

    def compute_moving_average(self):
        # 加权平均计算暂时存在问题 需要加入权重因子的影响（近端时间权重大，远端权重小）
        self.data_BCHAIN['moving_average'] = self.data_BCHAIN['value'].rolling(window=self.n_above_moving_average_BCHAIN).mean()
        self.data_GOLD['moving_average'] = self.data_GOLD['value'].rolling(window=self.n_above_moving_average_GOLD).mean()

    def compute_buy(self):
        # 0 表示卖，1 表示买入
        self.data_BCHAIN['buy_or_sell'] = 0
        self.data_GOLD['buy_or_sell'] = 0
        self.data_BCHAIN.loc[self.data_BCHAIN['grad'] > self.thr_BCHAIN and self.data_BCHAIN['value'] > self.data_BCHAIN['moving_average'] * self.rate_above_moving_average_BCHAIN, 'buy_or_sell'] = 1
        self.data_BCHAIN.loc[self.data_BCHAIN['value'] < self.data_BCHAIN['moving_average'] * self.rate_under_moving_average_BCHAIN, 'buy_or_sell'] = 1

    def compute_extreme_market(self):
        self.data_BCHAIN['Extreme Market'] = False
        self.data_GOLD['Extreme Market'] = False
        self.data_BCHAIN.loc[self.data_BCHAIN['grad'] - self.rate_Extreme_Market * self.data_BCHAIN['grad'].rolling(self.size_window_Extreme_Market).mean() > 0, 'Extreme Market'] = True
        self.data_GOLD.loc[self.data_GOLD['grad'] - self.rate_Extreme_Market * self.data_GOLD['grad'].rolling(self.size_window_Extreme_Market).mean() > 0, 'Extreme Market'] = True