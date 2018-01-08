import requests
import math
import sqlite3
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import ccxt
import datetime

def create_sqlite_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('CREATE TABLE results (timestamp real, instrument text, from_price read, to_price real, profit real)')
    conn.commit()
    conn.close()
#%%
def get_euro_rate():
    ## get fiat exchange rates in EUR
    r=requests.get('https://api.fixer.io/latest?base=EUR')
    return r.json()['rates']

def get_kraken_to_btcmarkets():
    #compare prices for EUR and AUD markets on Kraken and BTCmarkets
    #returns a dataframe of timestamp, token, profit
    
    # find all the AUD markets on btcmarkets
    btcmarkets = ccxt.btcmarkets()
    btcmarkets.load_markets()
    for name, marketdata in btcmarkets.markets.items():
        if marketdata['quote'] == 'AUD':
            marketdata.update(btcmarkets.fetch_ticker(name)['info'])
    btcmarketsdf = pd.DataFrame(btcmarkets.markets).T.query('quote == "AUD"')
    # Now do the same thing on Kraken
    kraken = ccxt.kraken()
    #here we can actually get all markets and all tickers in just 2 calls
    krakendf = pd.DataFrame(kraken.load_markets()).T
    #we are mainly interested in the bid price since we just need to make the highest bid
    krakenbids = pd.Series({name: tickerdata['bid'] for name, tickerdata in kraken.fetch_tickers().items()})
    krakendf['bid']=krakenbids
    # only look at euro ones
    krakendf = krakendf.query('quote=="EUR" and active==True and darkpool==False')
    #withdrawal_fees = {'BTC':0.001}
    ## Okay how can we sell these tokens at btcmarkets?
    df = pd.merge(btcmarketsdf, krakendf, on='base')
    rates = get_euro_rate()
    df['eurobid'] = df.bestBid / rates['AUD']
    df['profit'] = (df.eurobid / df.bid) * (1 - df.taker_y) * (1 - df.taker_x) - 1
    df = df.sort_values('profit', ascending=False)
    #assign all these results the same timestamp because they are close enough
    df.timestamp = pd.to_datetime(df.timestamp.max(),unit='s')
    results_df = df[['timestamp','instrument','bid','eurobid','profit']]
    results_df.columns = ['timestamp','instrument','from_price','to_price','profit']
    return results_df

def save_results_table(results): 
    #just a small table of current profits as a text output
    with open('public/results.txt','w') as outfile:
        for row in results.itertuples():
            outfile.write(f'{row.instrument} {row.profit:0.2%}\n')
        outfile.write(f'({results.iloc[0].instrument}<>{results.iloc[-1].instrument}) {results.iloc[0].profit-results.iloc[-1].profit:0.2%}\n')
#%%
def append_results_db(results):
    conn = sqlite3.connect('database.db')
    table_df = pd.read_sql_query('select * from results',conn)
    dt = pd.to_datetime(table_df.timestamp)
    table_df = table_df[datetime.datetime.now() - dt < datetime.timedelta(days=3)]
    complete_df = pd.concat([table_df, results], ignore_index=True)
    complete_df.to_sql('results',conn,if_exists='replace', index=False)
    conn.close()

#%%
def chart_results():
    #read results from flat file and make a chart
    sns.set_style('darkgrid')
    sns.set_context('talk')
    conn = sqlite3.connect('database.db')
    table_df = pd.read_sql_query('select * from results',conn)
    conn.close()
    rdf = table_df.pivot(index='timestamp',columns='instrument',values='profit')
    rdf.index = pd.DatetimeIndex(rdf.index)
    #sort the columns so the current-highest-profit one is first
    rdf = rdf[rdf.iloc[-1].sort_values(ascending=False).index]
    #name the columns with the current profit value
    rdf.columns = [f'{name} {profit:0.2%}' for name, profit in rdf.iloc[-1].iteritems()]
    ax = rdf.plot(figsize=(12,8), title='Arbitrage Kraken to BTC Markets')
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.1%}'.format))
    fig = ax.get_figure()
    fig.savefig('public/chart.png')
    return ax
#%%
def chart_separate():
    conn = sqlite3.connect('database.db')
    table_df = pd.read_sql_query('select * from results',conn)
    table_df.index = pd.DatetimeIndex(table_df.timestamp)
    conn.close()
    instruments = sorted(table_df.instrument.unique())
    fig, axes = plt.subplots(math.ceil(len(instruments)/3),3, sharex=True, sharey=False, figsize=(18,10))
    for instrument, ax in zip(instruments, axes.flat):
        table_df[table_df.instrument==instrument][['from_price','to_price']].plot(ax=ax, legend=False)
        table_df[table_df.instrument==instrument].profit.plot(ax=ax, secondary_y=True, style='k', alpha=0.5)
        ax.set_title(instrument)
    plt.tight_layout()
    plt.savefig('public/subplots.png')
#%%
def refresh():
    ## This code should run every 5 minutes or so
    results = get_kraken_to_btcmarkets()
    save_results_table(results)
    append_results_db(results)
    chart_results();
    chart_separate()

if __name__ == "__main__":
    refresh()

#%% Analyse opportunities
#import scipy.ndimage
#conn = sqlite3.connect('database.db')
#table_df = pd.read_sql_query('select * from results',conn)
#conn.close()
#table_df.timestamp = pd.to_datetime(table_df.timestamp)
#rdf = table_df.pivot(index='timestamp',columns='instrument',values='profit')
#rdf.index = pd.DatetimeIndex(rdf.index)    
#threshold = 0.1
##find contiguous regions above threshold
#regions = scipy.ndimage.measurements.label(rdf>=threshold,[[0,1,0],[0,1,0],[0,1,0]])[0]   
#regions = pd.DataFrame(regions, index=rdf.index, columns=rdf.columns)
##stack this so we can use groupby
#regions = regions.stack().reset_index(name='region')
##%%
##calculate the time span for each region
#region_lengths = regions[regions.region>0].groupby(['instrument','region']).apply(lambda region: region.iloc[-1].timestamp - region.iloc[0].timestamp)
#region_samples = regions[regions.region>0].groupby(['instrument','region']).size()
##cdf = pd.DataFrame((region_lengths, region_samples))
#cdf = pd.concat((region_lengths, region_samples), axis=1,names=['length','samples'])
#cdf.columns=['length','samples']
#cdf.length = cdf.length + pd.Timedelta('5 minutes')
#cdf['sample_rate'] = cdf.length / cdf.samples
#cdf[cdf.sample_rate>pd.Timedelta('7 minutes')]
##actually there's so few of these bad regions that it doesn't matter
##%%
#cdf['minutes']=cdf.length/pd.Timedelta('1 minute')
#print('Median time above 10% in the 6 day dataset')
#print(cdf.minutes.unstack(level=0).median())
#print('Median time above 10% when already seen for 10 minutes')
#print(cdf[cdf.minutes>=10].minutes.unstack(level=0).median())
#print('How many times it was above 10% for more than 20 minutes')
#print(cdf[cdf.minutes>=20].minutes.unstack(level=0).count())
#%% Simulations
#transaction_time_minutes = {'ETH': 6,
#                            'LTC': 30,
#                            'BTC': 60,
#                            'BCH': 60,
#                            'ETC': 28,
#                            'XRP': 3}
#from collections import defaultdict
#
#funds = defaultdict(float)
#funds['EUR'] = 1000
#eurobalances = []
#min_profit_to_aus = 0.09
#max_loss_to_europe = 0.05
#print('Starting with',funds['EUR'], 'EUR')
#disallow = ['BTC','BCH']
#print('Banned coins:',disallow)
#print('Outbound and inbound thresholds:',min_profit_to_aus, max_loss_to_europe)
#state = 'buying_europe'
#for row in table_df.itertuples():
#    if state == 'buying_europe':
#        if row.profit>min_profit_to_aus and row.instrument not in disallow:
#            #buy this coin
##            print(f'Buying {row.instrument} for {row.from_price}')
#            funds[row.instrument]+=funds['EUR']/row.from_price * (1-0.0026)
#            funds['EUR']=0
#            sell_time = row.timestamp + pd.Timedelta(minutes = transaction_time_minutes[row.instrument])
#            state = 'selling_aus'
#    elif state == 'selling_aus':
#        if (row.timestamp>=sell_time) and (funds[row.instrument]>0):
##            print(f'Selling {row.instrument} for {row.to_price}')
#            funds['AUD_EUR'] += funds[row.instrument] * row.to_price * (1 - 0.007)
#            funds[row.instrument] = 0
#            state = 'buying_aus'
#    elif state == 'buying_aus':
#        if row.profit<max_loss_to_europe and row.instrument not in disallow:
#            #buy this coin
##            print(f'Buying {row.instrument} for {row.to_price}')
#            funds[row.instrument]+=funds['AUD_EUR']/row.to_price  * (1 - 0.007)
#            funds['AUD_EUR']=0
#            sell_time = row.timestamp + pd.Timedelta(minutes = transaction_time_minutes[row.instrument])
#            state = 'selling_europe'
#    elif state == 'selling_europe':
#        if (row.timestamp>=sell_time) and (funds[row.instrument]>0):
##            print(f'Selling {row.instrument} for {row.from_price}')
#            funds['EUR'] += funds[row.instrument] * row.from_price  * (1-0.0026)
#            funds[row.instrument] = 0
#            state = 'buying_europe'
##            print(f'Cycle complete. We have {funds["EUR"]} euros.')
#            eurobalances.append(funds['EUR'])
#result = pd.Series(eurobalances)
#print(f'Complete after {len(result)} cycles. Final balance {result.iloc[-1]:.2f} euro')
#gains = result.diff()/result
#print(f'Average gain per cycle was {gains.mean():.1%}, max {gains.max():.1%}, min {gains.min():.1%}')
#print(f'Gains were negative on {len(gains[gains<0])} cycles')
#result.plot();
##%% Bettter bot that takes the best coin for each trade
#funds = defaultdict(float)
#funds['EUR'] = 1000
#trades=[]
#min_profit_to_aus = 0.07
#max_loss_to_europe = 0.05
#min_cycle_profit = 0.04
#import random
#min_random_delay_mins = 1
#max_random_delay_mins = 300
#print('Starting with',funds['EUR'], 'EUR')
#disallow = ['BTC','BCH']
#print('Banned coins:',disallow)
#print('Outbound and inbound thresholds:',min_profit_to_aus, max_loss_to_europe)
#state = 'buying_europe'
#holding = None
#for timestamp, rows in table_df[table_df.timestamp>pd.Timestamp('2016-12-24')].groupby('timestamp'):
#    if state == 'buying_europe':
#        #find the coin with the best profit
#        goodcoins = rows[~rows.instrument.isin(disallow)].sort_values('profit',ascending=False)
#        #can we profit from a cycle?
#        topcoin = goodcoins.iloc[0]
#        if(goodcoins.iloc[0].profit - goodcoins.iloc[-1].profit > min_cycle_profit):# or (topcoin.instrument=='XRP' and topcoin.profit>min_profit_to_aus):
##        if topcoin.profit>min_profit_to_aus:
#            #buy this coin
##            print(f'Buying {topcoin.instrument} for {topcoin.from_price}')
#            funds[topcoin.instrument]+=funds['EUR']/topcoin.from_price * (1-0.0026)
#            funds['EUR']=0
#            holding = topcoin.instrument
#            sell_time = timestamp + pd.Timedelta(minutes = transaction_time_minutes[topcoin.instrument] + random.randint(min_random_delay_mins,max_random_delay_mins))
#            state = 'selling_aus'
#    elif state == 'selling_aus':
#        if timestamp>=sell_time:
#            #sell this coin
#            sell_price = rows[rows.instrument==holding].iloc[0].to_price
##            print(f'Selling {holding} for {sell_price}')
#            funds['AUD_EUR'] += funds[holding] * sell_price * (1 - 0.007)
#            trades.append({'timestamp':timestamp,'instrument':holding,'direction':state,'balance':funds['AUD_EUR']})
#            funds[holding] = 0
#            state = 'buying_aus'
#            holding = None
#    elif state == 'buying_aus':
#        #sort the coins from high to low again
#        goodcoins = rows[~rows.instrument.isin(disallow)].sort_values('profit',ascending=False)
#        topcoin = goodcoins.iloc[-1] #this one has the lowest price difference
#        if (goodcoins.iloc[0].profit - goodcoins.iloc[-1].profit > min_cycle_profit):# or (topcoin.instrument=='XRP' and topcoin.profit<max_loss_to_europe):
##        if topcoin.profit<max_loss_to_europe:
#            #buy this coin
##            print(f'Buying {topcoin.instrument} for {topcoin.to_price}')
#            funds[topcoin.instrument]+=funds['AUD_EUR']/topcoin.to_price  * (1 - 0.007)
#            funds['AUD_EUR']=0
#            sell_time = timestamp + pd.Timedelta(minutes = transaction_time_minutes[topcoin.instrument]+random.randint(min_random_delay_mins,max_random_delay_mins))
#            state = 'selling_europe'
#            holding = topcoin.instrument
#    elif state == 'selling_europe':
#        if timestamp>=sell_time:
#            sell_price = rows[rows.instrument==holding].iloc[0].from_price
##            print(f'Selling {holding} for {sell_price}')
#            funds['EUR'] += funds[holding] * sell_price  * (1-0.0026)
#            funds[holding] = 0
#            trades.append({'timestamp':timestamp,'instrument':holding,'direction':state,'balance':funds['EUR']})
#            state = 'buying_europe'
#            holding = None
##            print(f'Cycle complete. We have {funds["EUR"]} euros.')
#            #eurobalances.append(funds['EUR'])
#trades = pd.DataFrame(trades)
#trades['gain'] = trades.balance.diff()/trades.balance
#trades.loc[0,'gain'] = trades.iloc[0].balance/1000-1
#
##ax = sns.countplot(data=trades, x='instrument',hue='direction')
##ax.set_title('Frequency of coin usage')
##plt.figure()
##ax = sns.barplot(data=trades[trades.direction=='selling_aus'], x='instrument',y='gain')
##ax.set_title('Profitability of outward coins')
##plt.figure()
##ax = sns.barplot(data=trades[trades.direction=='selling_europe'], x='instrument',y='gain')
##ax.set_title('Loss on inward coins')
#print('Mean gains for each coin')
#print(trades.groupby(['direction','instrument']).gain.mean().unstack())
#print()
#print('Frequency of coin usage')
#print(trades.groupby(['direction','instrument']).size().unstack())
#print()
#
#cycles=pd.merge(trades[trades.direction=='selling_aus'].reset_index(),trades[trades.direction=='selling_europe'].reset_index(),left_index=True,right_index=True)
#cycles['overall_gain'] = cycles.gain_x + cycles.gain_y
#print(f'Complete after {len(cycles)} cycles. Final balance {cycles.iloc[-1].balance_y:.2f} euro')
#print(f'Average gain per cycle was {cycles.overall_gain.mean():.1%}, max {cycles.overall_gain.max():.1%}, min {cycles.overall_gain.min():.1%}')
#print(f'Gains were negative on {sum(cycles.overall_gain<0)} cycles')
##cycles[['balance_x','balance_y']].plot();
#trades.set_index('timestamp',inplace=True)
##trades.balance.plot();
#cycles.set_index('timestamp_y').balance_y.plot(style='-x');
