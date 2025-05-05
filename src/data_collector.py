import pandas as pd
import numpy as np
import re
# import requests
import multiprocessing
from scipy.stats import spearmanr
import pandas as pd
from datetime import datetime, timedelta
import polars as pl
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import re
import warnings
from config import Data_downloader_window,OHCL_range_window,ARMM_Threshold, getLiveNews,getSymbols,getSpotData,root_path
# from db_conn import getCloudConnection
warnings.filterwarnings("ignore")



# engine = getCloudConnection() 

# Minute data downloader
def dataDownloaderParallel(symbol,scrapped_minute_data):
    """This function will download market minute data from vista's database api in parallel.
    Args:
        symbol (str): Symbol for the minute data
    Returns:
        pd.DataFrame: Dataframe consists of spot prices for the given symbol
    """
    window_size  = Data_downloader_window
    range_peroid = OHCL_range_window
    
    if len(scrapped_minute_data)>0:

        # Calculating the average of OHLC columns
        scrapped_minute_data['avg_ohlc'] = (scrapped_minute_data['OPEN'] + scrapped_minute_data['HIGH'] +
                                        scrapped_minute_data['LOW'] + scrapped_minute_data['CLOSE']) / 4
        
        # Getting the first avg_ohlc value
        windows_first_avg_value = scrapped_minute_data.iloc[0]['avg_ohlc']
        scrapped_minute_data['ASSET'] = symbol
        # If given symbol is Nifty 50's then calculating opportunities
        if symbol == "NIFTY 50.NSE_IDX":
            
            # Calculating the HIGH minus LOW movement
            scrapped_minute_data['h_l_movement'] = scrapped_minute_data['HIGH'] - scrapped_minute_data['LOW']
            # Calculating the HIGH minus CLOSW movement
            scrapped_minute_data['h_c_movement'] = abs(scrapped_minute_data['HIGH'] 
                                                    - scrapped_minute_data['CLOSE'].shift())
            # Calculating the LOW minus CLOSE movement
            scrapped_minute_data['l_c_movement'] = abs(scrapped_minute_data['LOW'] 
                                                    - scrapped_minute_data['CLOSE'].shift())
            # Calculating the maximum movement
            scrapped_minute_data['max_movement'] = scrapped_minute_data[['l_c_movement',
                                                                        'h_l_movement',
                                                                        'h_c_movement']].max(axis=1)
            # Calculating the Average Rolling Maximum Movements
            scrapped_minute_data['ARMM']      = scrapped_minute_data['max_movement'].rolling(window = range_peroid).mean()
            # Creating a minimun threshold of movement
            scrapped_minute_data['threshold'] = scrapped_minute_data['ARMM']
            # Initializing the movement threshold 
            scrapped_minute_data['ARMM_threshold'] = ARMM_Threshold
            for i in range(1,len(scrapped_minute_data)):
                if (scrapped_minute_data['CLOSE'].iloc[i] > scrapped_minute_data['ARMM_threshold'].iloc[i-1]) and (scrapped_minute_data['CLOSE'].iloc[i-1] > scrapped_minute_data['ARMM_threshold'].iloc[i-1] ):
                    scrapped_minute_data['ARMM_threshold'].iloc[i] = float(max(
                                                                    scrapped_minute_data['ARMM_threshold'].iloc[i-1],
                                                                    scrapped_minute_data['CLOSE'].iloc[i] - scrapped_minute_data['threshold'].iloc[i]
                                                                    )
                                                                        )
                elif (scrapped_minute_data['CLOSE'].iloc[i] < scrapped_minute_data['ARMM_threshold'].iloc[i-1]) and (scrapped_minute_data['CLOSE'].iloc[i-1] < scrapped_minute_data['ARMM_threshold'].iloc[i-1] ):
                    scrapped_minute_data['ARMM_threshold'].iloc[i] = float(min(
                                                                    scrapped_minute_data['ARMM_threshold'].iloc[i-1],
                                                                    scrapped_minute_data['CLOSE'].iloc[i] + scrapped_minute_data['threshold'].iloc[i]
                                                                    )
                                                                        )
                elif scrapped_minute_data['CLOSE'].iloc[i] > scrapped_minute_data['ARMM_threshold'].iloc[i-1]:
                    scrapped_minute_data['ARMM_threshold'].iloc[i] = float(scrapped_minute_data['CLOSE'].iloc[i] - scrapped_minute_data['threshold'].iloc[i])
                else:
                    scrapped_minute_data['ARMM_threshold'].iloc[i] = float(scrapped_minute_data['CLOSE'].iloc[i] + scrapped_minute_data['threshold'].iloc[i])
            # Position Opportunity Calculation
            scrapped_minute_data['above']         = np.where(scrapped_minute_data['CLOSE'] > scrapped_minute_data['ARMM_threshold'],1,0)
            scrapped_minute_data['below']         = np.where(scrapped_minute_data['ARMM_threshold'] > scrapped_minute_data['CLOSE'],1,0)
            scrapped_minute_data['opportunity_1'] = np.where((scrapped_minute_data['CLOSE'] > scrapped_minute_data['ARMM_threshold']) & (scrapped_minute_data['above']==1),1,0)
            scrapped_minute_data['opportunity_2'] = np.where((scrapped_minute_data['CLOSE'] < scrapped_minute_data['ARMM_threshold']) & (scrapped_minute_data['below']==1),1,0)
            # Removing the first instance of the window
            scrapped_minute_data = scrapped_minute_data[(1 - window_size):]    
            # Scaling the average OHLC
            scrapped_minute_data['sclaed_ohlc'] = ((scrapped_minute_data['avg_ohlc'] - windows_first_avg_value) / windows_first_avg_value) * 100
            # Dropping the unnecessary columns
            scrapped_minute_data.drop(['DATETIME', 'avg_ohlc','OPEN', 'HIGH', 'LOW', 'CLOSE', 'h_l_movement', 'h_c_movement', 'l_c_movement', 'max_movement', 'ARMM', 'threshold',
                                    'ARMM_threshold', 'above', 'below'], axis=1, inplace=True)
            return pd.DataFrame(scrapped_minute_data).reset_index(drop= True)
        else:
            # Removing the first instance of the window
            scrapped_minute_data = scrapped_minute_data[(1 - window_size):]
            # Scaling the average OHLC
            scrapped_minute_data['sclaed_ohlc']   = ((scrapped_minute_data['avg_ohlc'] - windows_first_avg_value) / windows_first_avg_value) * 100
            # Calculating the positional opportunities (Do it later for other stocks)
            scrapped_minute_data['opportunity_1'] = ''  # Currently assigning it to blank
            scrapped_minute_data['opportunity_2'] = ''  # Currently assigning it to blank
            scrapped_minute_data.drop(['VOLUME', 'OI', 'avg_ohlc','OPEN', 'HIGH', 'LOW', 'CLOSE'], axis = 1, inplace= True)
            return pd.DataFrame(scrapped_minute_data).reset_index(drop= True)



# Parallel datadownloader function
def fetchInsertCalculateDataParallel(symbol,date,trading_time):
    """Download, insert, and calculate minute data for a given symbol.
    Args:
        symbol (str): Symbol for the minute data

    Returns:
        pd.DataFrame: Dataframe consists of spot prices for the given symbol
    """
    try:
        scrapped_minute_data = getSpotData(symbol,date,trading_time)
        minute_data = dataDownloaderParallel(symbol,scrapped_minute_data)
        return minute_data
    except Exception as error:
        print("dataDownloaderParallel",error)
        pass
    

# Reading the nifty keyword/symbol dataframe
symbl_df = pd.read_csv(root_path+"data/nifty_companies.csv")

# Symbol Matching function
def findMatchingSymbols(args):
    text, keyword, symbol = args
    if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
        return symbol
    return None

# Function to find list of symbols inside a text in parallel
def findMatchingSymbolsParallel(text, df = symbl_df):
    with ThreadPoolExecutor() as executor:
        args_list = [(text, kw, sym) for kw, sym in zip(df['Company Name'], df['Symbol'])]
        matching_symbols_list = list(executor.map(findMatchingSymbols, args_list))

    matching_symbols = set(symbol for symbol in matching_symbols_list if symbol is not None)
    return list(matching_symbols)

# News Fetching function 
def fetchNews(date:str,trading_time,window_size:int,df_live_news,logger)-> pd.DataFrame:
    """This function collects the news data from the news api url for a given time interval

    Args:
        date (str) (yyyy-mm-dd): Date for which we want to fetch the news
        window_size (int): Number of minutes we want to check the news data from now

    Returns:
        pd.DataFrame: Fetched news based on the window_size 
    """

    try:
        # News API URL
        news_api_url   = f"https://api.vistaintelligence.ai/requesthandler/v1/news/getNewsTweetsForADate?date={date}&type=news"

        # # Setting the lookback time threshold
        # time_threshold2 = pd.to_datetime(date+' '+trading_time) - timedelta(minutes = window_size*3)
        # # Collecting response from the API
        # rspns        = requests.get(news_api_url).json()
        # # Creating the lazy dataframe
        # lazy_df      = pl.LazyFrame(rspns)

        # # Creating a query to load data
        # lazy_query   = (lazy_df.with_columns(
        #                         time_stamp = (
        #                             pl.col('datePublished').map_elements(lambda x : x[:10]) + ' ' + pl.col('datePublished').map_elements(lambda x : x[11:19])
        #                             ).str.to_datetime(),
        #                         news = (
        #                             pl.col('title') + ' ' + pl.col('summary')
        #                             )
        #                         ).filter(
        #                             (pl.col('time_stamp') >= time_threshold2) & ((pl.col('time_stamp') <= pd.to_datetime(date+" "+trading_time) - timedelta(minutes= 10)))
        #                             ).select('time_stamp','news','sentiment'))
        
        # start_time = datetime.now()
        # # Collecting the news
        # feedsense_news           = lazy_query.collect().to_pandas()
        # feedsense_news['symbol'] = feedsense_news['news'].apply(findMatchingSymbolsParallel)
        # # Concating the two dataframesinto one
        # output_news = pd.concat([feedsense_news,df_live_news],ignore_index= True)
        return df_live_news
    except Exception as error:
        logger.info(f"error in fetch news: {error}")
        pass


# Correlation calculator
def correlationCalculatorParallel(args):
    try:
        symbol, latest_nifty_percentage_movement, market_movement_dataset = args
        temp_minute_changes = market_movement_dataset[market_movement_dataset['ASSET'] == str(symbol)]['sclaed_ohlc'].tolist()
        # print("temp_minute_changes:", temp_minute_changes)
        temp_minute_changes_df = market_movement_dataset[market_movement_dataset['ASSET'] == str(symbol)]
        # print("temp_minute_changes_df:", temp_minute_changes_df)
        temp_time = temp_minute_changes_df.iloc[-1]['TIME']
        temp_date = temp_minute_changes_df.iloc[-1]['DATE']
        temp_corr = spearmanr(latest_nifty_percentage_movement, temp_minute_changes)[0]
        output_corr_df = pd.DataFrame({"DATE": [temp_date],"TIME": [temp_time],"ASSET": [symbol],"Corr": [temp_corr]})
        return output_corr_df
    except Exception as error:
        # print("error in correlationCalculatorParallel",error)
        pass

def parallelMarketDataCollector(date,trading_time,logger, nifty_data):
    try:
        stock_symbols = ['ADANIENT.NSE','ADANIPORTS.NSE','APOLLOHOSP.NSE','ASIANPAINT.NSE','AXISBANK.NSE','BAJAJ-AUTO.NSE',
                         'BAJFINANCE.NSE','BAJAJFINSV.NSE','BPCL.NSE','BHARTIARTL.NSE','BRITANNIA.NSE','CIPLA.NSE','COALINDIA.NSE','DRREDDY.NSE',
                         'EICHERMOT.NSE','GRASIM.NSE','HCLTECH.NSE','HDFCBANK.NSE','HDFCLIFE.NSE','HEROMOTOCO.NSE','HINDALCO.NSE','HINDUNILVR.NSE',
                        'ICICIBANK.NSE','ITC.NSE','INDUSINDBK.NSE','INFY.NSE','JSWSTEEL.NSE','KOTAKBANK.NSE','LT.NSE','M&M.NSE','MARUTI.NSE','NTPC.NSE',
                        'NESTLEIND.NSE','ONGC.NSE','POWERGRID.NSE','RELIANCE.NSE','SBILIFE.NSE','SHRIRAMFIN.NSE','SBIN.NSE','SUNPHARMA.NSE','TCS.NSE',
                        'TATACONSUM.NSE','TATAMOTORS.NSE','TATASTEEL.NSE','TECHM.NSE','TITAN.NSE','TRENT.NSE','ULTRACEMCO.NSE','WIPRO.NSE']#'BEL.NSE',
        # Creating a blank pandas dataframe
        minute_dataframe = pd.DataFrame()

        # Number of processes to use (adjust as needed)
        num_processes = multiprocessing.cpu_count()
        output = [(i,date,trading_time) for i in list(stock_symbols)]

        # Create a Pool of worker processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Apply the function in parallel using the Pool's map function
            try:
                results = pool.starmap(fetchInsertCalculateDataParallel,output)   
            except Exception as error:
                pass

        # print("results:", results)

        # Concatenate the results
        minute_dataframe = pd.concat(results, ignore_index=True).drop_duplicates()
        # Getting latest nifty returns
        nifty_movements = nifty_data['sclaed_ohlc'].tolist()  

        # Create a list of arguments for correlation_calculator_parallel
        correlation_args = [(symbol, nifty_movements, minute_dataframe) for symbol in stock_symbols]
        # print(correlation_args)

        # Create a Pool of worker processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Apply the function in parallel using the Pool's map function
            response = pool.map(correlationCalculatorParallel, correlation_args)

        # Concatenate the results
        correlation_dataframe = pd.concat(response, ignore_index=True).drop_duplicates()
        # print("""
        #         correlation_dataframe """,correlation_dataframe)
        return correlation_dataframe
    except Exception as error:
        logger.info(f"Exception in parallel Data Collector {trading_time}: {error}")
        pass


