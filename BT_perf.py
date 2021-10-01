import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import datetime
import ast


class DataMaker:
    """This class reads and creates new features for Report"""

    def __init__(self, path):
        self.path = path
        self.data = self.update_data()
        self.symbols = self.get_all_symbols()

        # print('__ Data Loaded __')

    def update_data(self):
        # print('dataframe has been reversed')
        data = pd.read_excel(self.path)[::-1]  # df's time is ascending, so we should reverse it
        data['Date(UTC)'] = pd.to_datetime(data['Date(UTC)'],
                                           format="%Y-%m-%d %H:%M:%S")  # Changing date column to datetime.
        data['TimeStamp'] = data['Date(UTC)'].astype('int64')
        return data

    def get_all_symbols(self):
        return list(self.data.Symbol.value_counts().index)

    def sub_data(self, pair):
        return self.data[self.data.Symbol == pair].copy()


class Portfolio:
    """This class assumes a portfolio and all other configurations"""

    def __init__(self, initial_inv, leverage, pair):
        self.initial_inv = initial_inv
        self.leverage = leverage
        self.pair = pair
        # print('__ Portfolio Made __')


class FeatureMaker:
    """This class makes new features for sensitivity analysis
    and risk analysis"""

    def __init__(self, asset_data):
        self.asset_data = asset_data
        self.start, self.end = self.asset_data['Date(UTC)'].iloc[0], self.asset_data['Date(UTC)'].iloc[-1]
        self.pair = self.asset_data.Symbol.iloc[0]
        self.holding, self.holding_time, self.profits, \
            self.profits_time, self.avg_grid_profits, self.grid_end_time = [], [], [], [], [], []
        self.Num_grid_pos, self.Num_transactions, \
            self.wins, self.draws, self.losses, \
            self.draws_win, self.draws_loss = 0, 0, 0, 0, 0, 0, 0
        self.add_all_features()
        print('__ Features Created __')

    def _add_pos(self):
        self.asset_data['pos'] = [1 if i == 'BUY' else -1 for
                                  i in self.asset_data.Side]

    def _cal_holding(self):
        self.asset_data['change_in_holding'] = \
            self.asset_data.Quantity * self.asset_data.pos
        self.asset_data['current_holding'] = \
            self.asset_data.change_in_holding.cumsum()
        self.asset_data['current_holding'] = [i if abs(i) > 0.000001 else 0 for i in self.asset_data['current_holding']]

    def is_pos_close(self, i):
        if self.asset_data.current_holding.iloc[i] == 0:
            return True
        else:
            return False

    @staticmethod
    def diff(start, end, reverse):
        if not reverse:
            return end - start
        if reverse:
            return start - end

    @staticmethod
    def diff_perc(start, end, reverse):
        if not reverse:
            return (end - start) / start
        if reverse:
            return (start - end) / end

    def is_reverse(self, i):
        if self.asset_data.pos.iloc[i] == 1:
            return True
        else:
            return False

    '''
    def convert_list_string_column(self, col):
        return self.asset_data[col][self.asset_data[col].notna()].apply(lambda s: eval(s))
    '''

    def get_end_start(self, col, i, j):
        end = self.asset_data[col].iloc[i]
        start = self.asset_data[col].iloc[i - j]
        return start, end

    def _do_selected_feature(self, temp_offset_feature_maker, offset, i, j, col):
        start, end = self.get_end_start(col, i, j)
        if offset == 'gross':
            temp_offset_feature_maker.append(self.diff(start, end, self.is_reverse(i)))
        elif offset == 'perc':  # This is for Price
            temp_offset_feature_maker.append(self.diff_perc(start, end, self.is_reverse(i)))
        elif offset == 'net':  # This is just for amount to calculate the net profit
            #start_fee, end_fee = self.get_end_start('Fee', i, j)
            FEE = 0.0004
            if not self.is_reverse(i):
                start_fee, end_fee = start * FEE, end * FEE
                temp_offset_feature_maker.append(self.diff_perc(start + start_fee, end - end_fee, self.is_reverse(i)))
            else:
                start_fee, end_fee = start * FEE, end * FEE
                temp_offset_feature_maker.append(self.diff_perc(start - start_fee, end + end_fee, self.is_reverse(i)))
        return temp_offset_feature_maker

    def _roll_on_diff(self, i, col, offset):
        temp_offset_feature_maker = []
        for j in range(1, i + 1):
            if not self.is_pos_close(i - j):
                temp_offset_feature_maker = self._do_selected_feature(temp_offset_feature_maker, offset, i, j, col)
            else:
                break
        return temp_offset_feature_maker

    def _set_offset_feature_maker(self, feature_temp_list, i, target):
        self.asset_data[target].iloc[i] = str(feature_temp_list)

    def _offset_feature_maker(self, target='holding_time', col='TimeStamp', offset='gross'):
        """ Calculates the differences of SELL/BUY features of each trade for different features"""
        self.asset_data[target] = np.nan
        for i in range(self.asset_data.shape[0]):
            if self.is_pos_close(i):
                temp_diff_list = self._roll_on_diff(i, col, offset)
                self._set_offset_feature_maker(temp_diff_list, i, target)
        # Convert back to list since the new columns where assigned as str in _set_offset_feature_maker
        # self.asset_data[col] = self.convert_list_string_column(self.asset_data[col])

    def _cal_profit_stat(self):
        for i in range(self.asset_data.shape[0]):
            if self.is_pos_close(i):
                temp_profits = []
                j = 1
                for profit in eval(self.asset_data.net_profit.iloc[i]):
                    temp_profits.append(profit)
                    # Profit by itself can bias our daily and cumulative return estimation since the investment is
                    # blocked inside each grid
                    self.profits.append(profit)
                    self.profits_time.append(self.asset_data.TimeStamp.iloc[i - j])
                    j += 1
                # Using geometric mean to calculate each grid profit
                self.avg_grid_profits.append(np.power(np.product(np.array(temp_profits) + 1), 1/len(temp_profits)) - 1)
                self.grid_end_time.append(self.asset_data.TimeStamp.iloc[i])

    def _cal_win_loss_draw(self):
        NANOSEC_HOUR = 3600000000000
        for i in range(self.asset_data.shape[0]):
            if self.is_pos_close(i):
                for profit, time in zip(
                        eval(self.asset_data.net_profit.iloc[i]), eval(self.asset_data.holding_time.iloc[i])
                ):
                    mask_profit = profit > 0
                    mask_draw = time > NANOSEC_HOUR
                    if mask_profit:
                        self.wins += 1
                    else:
                        self.losses += 1
                    if mask_draw: #This assumptionn is wrong
                        self.draws += 1
                        if mask_profit:
                            self.draws_win += 1
        self.draws_loss = self.draws - self.draws_win

    def _cal_num_features(self):
        self.Num_grid_pos = self.asset_data[self.asset_data.current_holding == 0].shape[0]
        self.Num_transactions = self.asset_data.shape[0]

    def save_csv(self):
        self.asset_data.to_csv(f'./Features Dataframes/{self.pair}_feature_result.csv')

    def add_all_features(self):
        self._add_pos()
        self._cal_holding()
        self._offset_feature_maker(target='holding_time', col='TimeStamp', offset='gross')
        self._offset_feature_maker(target='gross_price_diff', col='Price')
        self._offset_feature_maker(target='gross_quantity_diff', col='Quantity')
        self._offset_feature_maker(target='gross_amount_diff', col='Amount')
        self._offset_feature_maker(target='net_profit', col='Price', offset='net')
        self._cal_win_loss_draw()
        self._cal_num_features()
        self._cal_profit_stat()
        self.save_csv()


class ReportMaker:
    """This class illustrate the report"""

    def __init__(self, data_obj, portfolio_obj, feature_obj):
        self.obj = feature_obj
        self.VaR_99, self.VaR_95 = 0, 0
        self.make_rep()

    """This part is for helper functions"""
    def hist_pnl(self, title, ycol, xlabel, ylabel, name, vert_line):
        plt.figure(figsize=(20, 15))
        kwargs = dict(alpha=0.5, density=True, stacked=True)
        binBoundaries = np.linspace(-3, 7, 100)
        plt.hist(ycol, bins=binBoundaries, **kwargs)
        if vert_line:
            plt.axvline(self.VaR_95, c='red')
            plt.axvline(self.VaR_99, c='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=90)
        if not vert_line:
            plt.savefig(f'./Plots/PnL Distributions/{self.obj.pair}_profit_distribution.png')
        else:
            plt.savefig(f'./Plots/PnL Distributions/{self.obj.pair}_profit_distribution_with_{name}.png')

    def rooling_on_hist_pnl(self, title, xlabel, ylabel, rolling_type, xcol=None, ycol=None,
                            name=None, vert_line=False):
        if rolling_type == 'profits':
            ycol = self.obj.profits
        else:
            ycol = self.obj.asset_data[ycol]
        self.hist_pnl(title, ycol, xlabel, ylabel, name, vert_line)

    def set_VaR(self, method='simple'):
        profits = pd.Series(self.obj.profits).sort_values()
        if method == 'simple':
            self.VaR_99 = profits.quantile(0.01)
            self.VaR_95 = profits.quantile(0.05)
            print(f'VaR95 = {self.VaR_95}, VaR99 = {self.VaR_99}')

    def get_ES(self):
        pass

    def get_Abs_drawdown(self):
        pass

    def get_sharp_ratio(self):
        pass

    """This part is allocated to reports"""
    def perf_rep(self):
        """Most of the stats are based on grid profit"""
        output = pd.DataFrame()
        Num_days = ((self.obj.end - self.obj.start).days + (self.obj.end - self.obj.start).seconds/86400) #86400 is the number of seconds in a day
        cum_return = ((np.product(np.array(self.obj.avg_grid_profits) + 1)) - 1)
        daily_return = np.power(1 + cum_return, 1/Num_days) - 1
        annual_profit = daily_return * 365
        temp_dict = {'Pair': self.obj.pair, 'Num_grid_pos': self.obj.Num_grid_pos,
                     'Num_transactions': self.obj.Num_transactions, 'len_profits': len(self.obj.avg_grid_profits),
                     'Number of days': Num_days, 'Daily return': daily_return,
                     'Cum_profit': cum_return,
                     'Annual_cum_profit': annual_profit,
                     'Tot_profit': sum(self.obj.avg_grid_profits),
                     'Avg_profit': np.mean(self.obj.avg_grid_profits), 'Num_wins': self.obj.wins,
                     'Num_losses': self.obj.losses, 'Num_draws': self.obj.draws,
                     'Num_draw_wins': self.obj.draws_win, 'Num_draw_losses': self.obj.draws_loss,
                     'Backtest_started': self.obj.start, 'Backtest_ended': self.obj.end,
                     }
        print(temp_dict)
        output = output.append(temp_dict, ignore_index=True)
        return output

    def liquidation_rep(self):
        pass

    def make_rep(self):
        self.perf_rep()
        self.rooling_on_hist_pnl(title='PnL Distribution', xlabel='Net Return', ylabel='count', rolling_type='profits')
        self.set_VaR(method='simple')
        self.rooling_on_hist_pnl(title='PnL Distribution', xlabel='Net Return', ylabel='count', rolling_type='profits'
                                 , xcol=None, ycol=None, name='VaR99 and VaR99', vert_line=True
                                 )


# path for reading back-test data
PATH = './Export Trade History.xlsx'
INITIAL_INV, LEVERAGE = 1000, 1


def test(data_object, portfolio_obj, features_obj, report_obj):
    #print(data_object.data)
    print(portfolio_obj.pair)
    #print(features_obj.asset_data.columns)
    #print(features_obj.asset_data.head(20))
    return report_obj.perf_rep()


def main(address: object, initial_inv: object, leverage: object, pair: object, ALL_PAIRS) -> object:
    d_obj = DataMaker(address)
    port_obj = Portfolio(initial_inv, leverage, pair)
    feat_obj = FeatureMaker(d_obj.sub_data(port_obj.pair))
    rep_obj = ReportMaker(d_obj, port_obj, feat_obj)
    perf_df = pd.DataFrame()
    perf_df.append(test(d_obj, port_obj, feat_obj, rep_obj))
    return perf_df


if __name__ == "__main__":
    ALL_PAIRS = DataMaker(PATH).symbols
    perf_rep_df = pd.DataFrame()
    for PAIR in ALL_PAIRS:
        perf_rep = main(PATH, INITIAL_INV, LEVERAGE, PAIR, ALL_PAIRS)
        perf_rep_df.append(perf_rep)
    print(perf_rep_df.head())