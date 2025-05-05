import pandas as pd
from config import dictToDataframe, getAllNiftyData, getLiveNews, getNiftyData, insertSentimentSignal, isHoliday, isPresent, setup_logger, minuteList, to_sleep, updateMinuteAccurecy,tickConnection
from data_collector import dataDownloaderParallel, parallelMarketDataCollector
from signal_generator import signalGenerator
from datetime import datetime, timedelta
from accuracy_generator import minuteAccuracyGenerator
from my_streamlit_app import toggle_filter_fun, toggleFilterFun

    

def automate(date,logger):
    logger.info(f"""**********************************Current Date : {date}**********************************""")
    j_sentiment = {}
    j_sentiment_updated = {}
    j_accurecy = {}
    db_action = 'update'
    if isPresent(date) == False:
        db_action = 'insert'
    else:
        db_action = 'update'

    count = 0
    result = None
    nifty_df             = pd.DataFrame()
    sentiment_df         = pd.DataFrame()
    updated_sentiment_df = pd.DataFrame()
    j_pl                 = {}
    
    for trading_time in minuteList:
        try:
            time_threshold   = pd.to_datetime(date+' '+trading_time) - timedelta(minutes = 30)
            nifty_df         = getNiftyData(date,trading_time)
            stt_live_news    = getLiveNews(time_threshold,pd.to_datetime(date+' '+trading_time[:6]+'45'),logger)
            nifty_data       = dataDownloaderParallel('NIFTY 50.NSE_IDX', nifty_df)
            all_symbols_data = parallelMarketDataCollector(date, str(trading_time), logger, nifty_data)
            result           = signalGenerator(correlation_dataframe = all_symbols_data, opportunity_dataframe = nifty_data,news_dataframe = stt_live_news)

            if ((result != 'NA') and (result != None)):
                j_sentiment[str(nifty_data['TIME'].iloc[-1])] = result
                sentiment_df         = dictToDataframe(date,j_sentiment)
                updated_sentiment_df = toggle_filter_fun(sentiment_df)
                j_sentiment_updated  = toggleFilterFun(j_sentiment)

                db_action = insertSentimentSignal(date,trading_time,j_sentiment,j_sentiment_updated,j_accurecy,result,db_action,logger)
            else:
                logger.info(f"Model time: {trading_time} News Sentiment Signal : {result}")

            if count >= 15:
                try:
                    all_nifty_data = getAllNiftyData(date ,trading_time)
                    accrecy_dict, pl_dict = minuteAccuracyGenerator(updated_sentiment_df, all_nifty_data)
                    j_accurecy[trading_time] = accrecy_dict
                    j_pl[trading_time] = pl_dict
                    updateMinuteAccurecy(date, j_accurecy, j_pl, trading_time, logger)
                    
                    count = 0
                except Exception as err: 
                    logger.info(f"Accurecy error : {err}")
                    pass
            count = count+1
            
        except Exception as error:
            logger.info(f"Signal error : {error}")
            pass

        to_sleep(date,'minute_sleep',trading_time,logger)
        


    

current_date = '2025-03-27' #str(datetime.today().date())

logger = setup_logger(current_date)
if isHoliday(current_date) == False:
    automate(current_date, logger)
    tickConnection.close()
else:
    logger.info(f"{current_date} is Holiday")
