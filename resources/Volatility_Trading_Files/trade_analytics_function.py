#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: QuantInsti
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trade_level_analytics(round_trips, lot_size):

    # Calculate net premium
    round_trips['trade_wise_PnL'] = round_trips['trade_position'] * (round_trips['trade_exit_price'] - round_trips['trade_entry_price'])

    # Create a dataframe for storing trades
    trades = pd.DataFrame()

    # Groupby entry date
    trades_group = round_trips.groupby('trade_entry_date')

    # Group trades from round_trips
    trades['trade_entry_date'] = trades_group['trade_entry_date'].first()
    trades['trade_exit_date'] = trades_group['trade_exit_date'].first()

    # Calculate PnL for the strategy for 1 lot
    trades['pnl'] = trades_group.trade_wise_PnL.sum() * lot_size

    # Calculate turnover for trades
    trades['Turnover'] = (trades_group['trade_exit_price'].sum() + trades_group['trade_entry_price'].sum()) * lot_size

    # Calculate PnL after deducting trading costs and slippages
    trades['PnL_post_trading_costs_slippages'] = trades['pnl'] - trades['Turnover'] * (0.01)

    # Create dataframe to store trade analytics
    analytics = pd.DataFrame(index=['Strategy'])

    # Calculate total PnL
    analytics['Total PnL'] = round(trades.pnl.sum(),2)

    # Number of total trades
    analytics['Total Trades'] = int(len(trades))

    # Profitable trades
    analytics['Number of Winners'] = int(len(trades.loc[trades.pnl > 0]))

    # Loss-making trades
    analytics['Number of Losers'] = int(len(trades.loc[trades.pnl <= 0]))

    # Win percentage
    analytics['Win (%)'] = round(100 * analytics['Number of Winners'] / analytics['Total Trades'],2)

    # Loss percentage
    analytics['Loss (%)'] = round(100 * analytics['Number of Losers'] / analytics['Total Trades'],2)

    # Per trade profit/loss of winning trades
    analytics['Per Trade PnL of Winners'] = round(trades.loc[trades.pnl > 0].pnl.mean(), 2)

    # Per trade profit/loss of losing trades
    analytics['Per Trade PnL of Losers'] = round(np.abs(trades.loc[trades.pnl <= 0].pnl.mean()), 2)

    # Calculate profit factor
    analytics['Profit Factor'] = round((analytics['Win (%)'] / 100 * analytics['Per Trade PnL of Winners']) / (
            analytics['Loss (%)'] / 100 * analytics['Per Trade PnL of Losers']), 2)

    cum_pnl = round_trips.groupby('trade_exit_date')['pnl'].sum().cumsum()
    cum_pnl.plot()
    plt.title('Equity curve in $')
    plt.show()

    (cum_pnl - cum_pnl.cummax()).plot()
    plt.title('Drawdown in $')
    plt.show()

    return analytics.T