import numpy as np
import pandas as pd
import re

def fix_date_format(date_str):
    # 如果日期以 '200' 开头，则删除 '200'
    if isinstance(date_str, str) and re.match(r'^200\d{1}-\d{1,2}-\d{1,2}$', date_str):
        return date_str[3:]
    # 如果日期以 '20' 开头，则删除 '20'
    elif isinstance(date_str, str) and re.match(r'^20\d{2}-\d{1,2}-\d{1,2}$', date_str):
        return date_str[2:]
    else:
        # 如果不是上述两种情况，则保持原样
        return date_str

class Portfolio:
    def __init__(self):
        self.assets = {"bitcoin": 0, "gold": 0}
        self.cash = 1000  # 初始现金
        self.holding_period = {"bitcoin": 0, "gold": 0}  # 持有期计数
        self.no_buy_period = 0  # 无买入期计数

    def update_holding_period(self):
        for asset in self.holding_period:
            if self.holding_period[asset] > 0:
                self.holding_period[asset] -= 1

    def can_sell(self, asset):
        return self.holding_period[asset] == 0

    def can_buy(self):
        return self.no_buy_period == 0

    def update_no_buy_period(self):
        if self.no_buy_period > 0:
            self.no_buy_period -= 1

    def set_no_buy_period(self, days):
        self.no_buy_period = days

class Market:
    def __init__(self):
        # 读取数据
        self.data_BCHAIN = pd.read_csv("processed_BCHAIN-MKPRU.csv")
        self.data_GOLD = pd.read_csv("processed_data_gold.csv")
        #self.data_BCHAIN['date'] = self.data_BCHAIN['date'].apply(fix_date_format)
        #self.data_GOLD['date'] = self.data_GOLD['date'].apply(fix_date_format)
        self.data_BCHAIN = pd.DataFrame(self.data_BCHAIN)
        self.data_GOLD = pd.DataFrame(self.data_GOLD)
        
        # 确保日期列为datetime类型，并排序
        self.data_BCHAIN['date'] = pd.to_datetime(self.data_BCHAIN['date'], format='%Y-%m-%d')
        self.data_GOLD['date'] = pd.to_datetime(self.data_GOLD['date'], format='%Y-%m-%d')
        self.data_BCHAIN = self.data_BCHAIN.sort_values('date').reset_index(drop=True)
        self.data_GOLD = self.data_GOLD.sort_values('date').reset_index(drop=True)
        
        # 合并数据以确保同步
        self.data = pd.merge(self.data_BCHAIN, self.data_GOLD, on='date', suffixes=('_bitcoin', '_gold'), how='inner')
        
        # 参数初始化
        self.weight_factor_constant = 1  # 计算权重因子的常数 C
        self.t = 10  # 计算权重因子的范围
        self.C = self.weight_factor_constant
        self.weight_factor = self.C / (np.arange(1, self.t+1)**2)
        self.weight_factor = self.weight_factor/self.weight_factor.sum()
        
        self.n_moving_average = 10  # 移动平均窗口大小
        self.thr = 0.05  # 梯度阈值
        self.rate_above_moving_average = 1.05
        self.rate_under_moving_average = 0.80
        self.rate_extreme_market = 5
        self.size_window_extreme_market = 10
        self.holding_days = 5  # 持有期天数
        self.no_buy_days = 3  # 无买入期天数
        self.margin_L = 10  # 卖出利润门槛
        
        # 初始化信号列
        self.data['grad_bitcoin'] = 0
        self.data['grad_gold'] = 0
        self.data['moving_average_bitcoin'] = 0
        self.data['moving_average_gold'] = 0
        self.data['buy_or_sell_bitcoin'] = 0
        self.data['buy_or_sell_gold'] = 0
        self.data['Extreme_Market_bitcoin'] = False
        self.data['Extreme_Market_gold'] = False
        self.data['Profitability_bitcoin'] = 0
        self.data['Profitability_gold'] = 0
        
        # 确保 Gold 数据中包含 'is_interpolated' 列
        if 'is_interpolated' not in self.data.columns:
            raise ValueError("Gold 数据中必须包含 'is_interpolated' 列")
        
        # 记录交易过程
        self.trade_log = []

    def compute_gradient(self):
        self.data['grad_bitcoin'] = self.data['value_bitcoin'] - self.data['value_bitcoin'].shift(1)
        self.data['grad_gold'] = self.data['value_gold'] - self.data['value_gold'].shift(1)

    def compute_moving_average(self):
        self.data['moving_average_bitcoin'] = self.data['value_bitcoin'].rolling(window=self.n_moving_average).mean()
        self.data['moving_average_gold'] = self.data['value_gold'].rolling(window=self.n_moving_average).mean()

    def compute_buy_signals(self):
        # 动量和均值回归买入信号
        condition_BCHAIN = (self.data['grad_bitcoin'] > self.thr) & \
                           (self.data['value_bitcoin'] > self.data['moving_average_bitcoin'] * self.rate_above_moving_average)
        condition_BCHAIN |= (self.data['value_bitcoin'] < self.data['moving_average_bitcoin'] * self.rate_under_moving_average)

        condition_GOLD = (self.data['grad_gold'] > self.thr) & \
                         (self.data['value_gold'] > self.data['moving_average_gold'] * self.rate_above_moving_average)
        condition_GOLD |= (self.data['value_gold'] < self.data['moving_average_gold'] * self.rate_under_moving_average)

        # 仅当 'is_interpolated' 为 False 时才能买入/卖出黄金
        condition_GOLD &= ~self.data['is_interpolated']

        self.data['buy_or_sell_bitcoin'] = np.where(condition_BCHAIN, 1, -1)
        self.data['buy_or_sell_gold'] = np.where(condition_GOLD, 1, -1)

    def compute_extreme_market(self):
        # 计算极端市场条件
        self.data['Extreme_Market_bitcoin'] = (self.data['grad_bitcoin'] - 
            self.rate_extreme_market * self.data['grad_bitcoin'].rolling(self.size_window_extreme_market).mean() > 0)
        self.data['Extreme_Market_gold'] = (self.data['grad_gold'] - 
            self.rate_extreme_market * self.data['grad_gold'].rolling(self.size_window_extreme_market).mean() > 0)

    def compute_profitability(self):
        # 动量交易盈利性与梯度成正比
        self.data['Profitability_bitcoin'] = self.data['grad_bitcoin']
        self.data['Profitability_gold'] = self.data['grad_gold']
        
        # 均值回归盈利性与价格偏离均值的平方成正比
        deviation_BCHAIN = self.data['value_bitcoin'] - self.data['moving_average_bitcoin']
        deviation_GOLD = self.data['value_gold'] - self.data['moving_average_gold']
        self.data.loc[self.data['value_bitcoin'] < self.data['moving_average_bitcoin'] * self.rate_under_moving_average, 
                     'Profitability_bitcoin'] += (deviation_BCHAIN ** 2)
        self.data.loc[self.data['value_gold'] < self.data['moving_average_gold'] * self.rate_under_moving_average, 
                     'Profitability_gold'] += (deviation_GOLD ** 2)

    def compute_extreme_market_condition_logic(self, current_day):
        # 使用 Extreme_Market 列判断是否处于极端市场条件
        extreme_BCHAIN = self.data.loc[current_day, 'Extreme_Market_bitcoin']
        extreme_GOLD = self.data.loc[current_day, 'Extreme_Market_gold']
        return extreme_BCHAIN or extreme_GOLD

    def compute_sell_signals(self, portfolio, current_day):
        sell_signals = {}
        for asset in ['bitcoin', 'gold']:
            if asset == 'bitcoin':
                price = self.data.loc[current_day, 'value_bitcoin']
                buy_or_sell = self.data.loc[current_day, 'buy_or_sell_bitcoin']
            else:
                # 检查 'is_interpolated' 列，如果为 True 则不能交易黄金
                if self.data.loc[current_day, 'is_interpolated']:
                    sell_signals[asset] = False
                    continue
                price = self.data.loc[current_day, 'value_gold']
                buy_or_sell = self.data.loc[current_day, 'buy_or_sell_gold']
            
            if not portfolio.can_sell(asset):
                sell_signals[asset] = False
                continue
            
            current_price = price
            # 计算平均成本
            if portfolio.assets[asset] == 0:
                average_cost = 0
            else:
                # 计算持有资产的加权平均成本
                bought_prices = self.data.loc[:current_day][self.data.loc[:current_day, f'buy_or_sell_{asset}'] == 1][f'value_{asset}']
                if not bought_prices.empty:
                    average_cost = bought_prices.mean()
                else:
                    average_cost = 0
            # 计算卖出指标 S
            S = (1 - 0.01) * current_price - average_cost - self.margin_L  # 假设卖出手续费 fa = 1%
            sell_signals[asset] = S > 0
        return sell_signals

    def execute_trades(self, portfolio, current_day):
        # 获取当前日期
        current_date = self.data.loc[current_day, 'date']
        
        # 检查是否满足极端市场条件
        extreme_condition = self.compute_extreme_market_condition_logic(current_day)
        
        # 检查是否满足无买入条件
        if portfolio.no_buy_period > 0:
            can_buy = False
        else:
            can_buy = True
        
        # 如果极端市场条件被激活，执行相应逻辑
        if extreme_condition:
            # 不允许卖出，直到价格回归
            sell_signals = {"bitcoin": False, "gold": False}
            # 回归条件需要根据论文中的具体描述进一步实现，这里简化为不卖出
        else:
            # 计算卖出信号
            sell_signals = self.compute_sell_signals(portfolio, current_day)
        
        # 计算买入信号
        if can_buy:
            buy_signals = {
                "bitcoin": self.data.loc[current_day, 'buy_or_sell_bitcoin'] == 1,
                "gold": self.data.loc[current_day, 'buy_or_sell_gold'] == 1
            }
        else:
            buy_signals = {"bitcoin": False, "gold": False}
        
        # 初始化交易记录
        buy_bitcoin = 0
        sell_bitcoin = 0
        buy_gold = 0
        sell_gold = 0
        
        # 执行卖出操作
        for asset in ['bitcoin', 'gold']:
            if sell_signals[asset]:
                price = self.data.loc[current_day, f'value_{asset}']
                # 计算卖出金额
                if portfolio.assets[asset] > 0:
                    sell_amount = portfolio.assets[asset]
                    proceeds = (1 - 0.01) * sell_amount * price  # 假设卖出手续费 fa = 1%
                    portfolio.cash += proceeds
                    if asset == 'bitcoin':
                        sell_bitcoin = sell_amount
                    else:
                        sell_gold = sell_amount
                    portfolio.assets[asset] = 0
                    # 设置无买入期
                    portfolio.set_no_buy_period(self.no_buy_days)
        
        # 执行买入操作
        for asset in ['bitcoin', 'gold']:
            # 对黄金交易增加 'is_interpolated' 检查
            if asset == 'gold' and self.data.loc[current_day, 'is_interpolated']:
                continue  # 如果 'is_interpolated' 为 True，则跳过黄金交易
            
            if buy_signals[asset]:
                price = self.data.loc[current_day, f'value_{asset}']
                transaction_cost = 0.02 * portfolio.cash  # 假设买入手续费为2%
                available_cash = portfolio.cash - transaction_cost
                if available_cash > 0:
                    buy_amount = available_cash / price
                    portfolio.assets[asset] += buy_amount
                    portfolio.cash -= buy_amount * price + transaction_cost
                    if asset == 'bitcoin':
                        buy_bitcoin = buy_amount
                    else:
                        buy_gold = buy_amount
                    # 设置持有期
                    portfolio.holding_period[asset] = self.holding_days
        
        # 计算总资产价值
        total_value = portfolio.cash
        total_value += portfolio.assets['bitcoin'] * self.data.loc[current_day, 'value_bitcoin']
        total_value += portfolio.assets['gold'] * self.data.loc[current_day, 'value_gold']
        
        # 记录交易日志
        self.trade_log.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'buy_bitcoin': buy_bitcoin,
            'sell_bitcoin': sell_bitcoin,
            'buy_gold': buy_gold,
            'sell_gold': sell_gold,
            'holding_bitcoin': portfolio.assets['bitcoin'],
            'holding_gold': portfolio.assets['gold'],
            'cash': portfolio.cash,
            'total_value': total_value
        })

    def run_model(self):
        # 预处理
        self.compute_gradient()
        self.compute_moving_average()
        self.compute_buy_signals()
        self.compute_extreme_market()
        self.compute_profitability()
        
        # 初始化投资组合
        portfolio = Portfolio()
        
        # 遍历每一天的数据
        for current_day in range(1, len(self.data)):
            # 更新持有期和无买入期
            portfolio.update_holding_period()
            portfolio.update_no_buy_period()
            
            # 执行交易
            self.execute_trades(portfolio, current_day)
        
        # 转换交易日志为DataFrame
        trade_df = pd.DataFrame(self.trade_log)
        
        # 保存到Excel
        trade_df.to_excel("trade_log.xlsx", index=False)
        print("交易日志已保存到 'trade_log.xlsx'")
        print("Final Portfolio:", portfolio.assets)
        print("Final Cash:", portfolio.cash)

# 使用示例
if __name__ == "__main__":
    market = Market()
    market.run_model()
