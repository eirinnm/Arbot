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
    results.to_sql('results',conn,if_exists='append', index=False)
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
