import pandas as pd
from config import getNiftySpotData, insertUpdateFunction, isPresent, niftyDataFunction,root_path
from newsScrapper import fetchNewsForPreMarket
from post_market_sentiment import preMarketSentimentGeneration
from datetime import datetime
import logging
import math, time


def setupLogger(run_date):
    logger = logging.getLogger(f"{run_date}")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f'{root_path}datasave_logs/{run_date}_candle_data.log',mode='w')        
    formatter = logging.Formatter('At %(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def calculatePreMarketSentiment(curDate,logger):
    try:
        detailsDict = {"Date": curDate, "preMarketSentiment":None}
        dbAction='preMarketInsert'
        news_DF = fetchNewsForPreMarket(curDate)
        news_DF['DATETIME'] = pd.to_datetime(news_DF['datePublished'])
        news_DF['DATETIME'] = news_DF['DATETIME'].astype(str).apply(lambda x: x[:19])
        ans = preMarketSentimentGeneration(news_DF, news_column="summary", date_column="DATETIME")
        detailsDict.update({"preMarketSentiment": ans})
        if isPresent(curDate) == True:
            dbAction='preMarketUpdate'
        insertUpdateFunction(dbAction, detailsDict,logger, minute=str(datetime.now().time())[:8], sentimentValue=ans)
    except Exception as error:
        logger.info(f"Pre market sentiment error: {error}")


current_date = str(datetime.today().date())

logger = setupLogger(current_date)
logger.info(f"""--------------------------------------------------------------------------------------------------
            -------------------------------Premarket Sentiment for {current_date}--------------------------------
            --------------------------------------------------------------------------------------------------""")
calculatePreMarketSentiment(current_date,logger)


def to_sleep(trading_date, sleep_params, minute,logger, sleep_value=5):
    trade_minute = pd.to_datetime(minute)
    cur_minute = pd.to_datetime(datetime.now())
    total_minute =  cur_minute - trade_minute
    total_minute = math.floor(total_minute.total_seconds() / 60)
    if pd.to_datetime(trading_date) < pd.to_datetime(datetime.now().date()):
        return
    
    if total_minute == 1:
        if sleep_params == 'missing_sleep':
            sleep_value = sleep_value
        elif sleep_params == 'minute_sleep':
            sleep_value  = abs((60)-datetime.now().time().second)
        elif sleep_params == 'first_minute_sleep':
            sleep_value = ((180+sleep_value)-datetime.now().time().second)
            logger.info(f"{sleep_params} for {sleep_value}")
        time.sleep(sleep_value)
        

def StoreMinuteData(current_date):
    ohlc_dict = {}
    ohlc_action = 'insert'
    symbol = 'NIFTY 50.NSE_IDX'
    minuteList = pd.date_range("09:15:00", "15:30:00", freq="1min").time.astype(str)
    for minute in minuteList:
        try:
            current_minute_data = getNiftySpotData(current_date, minute)
            if len(current_minute_data) > 0:
                ohlc_dict.update({current_minute_data["TIME"].values[0]: [current_minute_data['OPEN'].values[0],current_minute_data['HIGH'].values[0],current_minute_data["LOW"].values[0],current_minute_data["CLOSE"].values[0]]})
                ohlc_action = niftyDataFunction(current_date,symbol,minute,ohlc_dict,ohlc_action,logger)
        except Exception as error:
            logger.info(f"Exception in minute_data: {error}")
            
        to_sleep(current_date,'minute_sleep',minute,logger)
            
StoreMinuteData(current_date)