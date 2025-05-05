# Database Credentials
import math
import pandas as pd
import json
import logging 
from datetime import datetime
import time
import sys

sys.path.append('/home/vista-ai') 
from db_conn import getLocalConnection, closeConnection,getTickConnection, getLiveConnection

# Minute data downloader configs
Data_downloader_window = 30
OHCL_range_window = 10
Sensitivity_Parameter = 1
NIFTY_50_Spot_Symbol = "NIFTY 50.NSE_IDX"
ARMM_Threshold = 0

# Necessary File paths
root_path = '/home/nilabja/Documents/nilabja/NLP/sentiment-live_feedsense/'
# FinBERT_Tone_Model_Path = root_path+'models/FinBERT-tone'
# weight_data_path = root_path+'data/news_weights.xlsx'
# classification_layer_path = root_path+"models/CNN_DNN_classifier"
# nifty_companies = root_path+'data/nifty_companies.csv'
# vosk_model_path = root_path+'models/vosk-model-en-us-0.42-gigaspeech'
# FinBERT_Tone_Model = 'FinBERT-FinVocab-Uncased'
# trained_classification_layer = "model.pth"
keyword_sheet = 'news_keywords'
filtering_keywords = 'Keywords'
# model_path = root_path+'model/vosk-model-small-en-us-0.15'

# Stopwords to ignore
stopwords_to_ignore = []
input_channels = 1
output_channels = 16
conv_kernal_size = 3
pooling_kernal_size = 2
dnn_input_size = 16
dnn_output_size = 64
input_size = 768
num_classes = 3
max_words_in_three_sentence = 80
padding = 'max_length'
class_mapping_of_sentiment = {2: -1, 0: 0, 1: 1}
news_class_weight_matrix =  [1,1,1,0.5,0.75]
minimum_weight_to_assign =  0.4
minuteList = pd.date_range("09:43:45", "15:29:45", freq="1min").time.astype(str)

def setup_logger(run_date):
    logger = logging.getLogger(f"{run_date}")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f'{root_path}logs/{run_date}_sentiment.log',mode='w')        
    formatter = logging.Formatter('At %(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

#---------------------------------------------------------------------------#
tickConnection  = getTickConnection()
def reconnect(conn):
    if conn.closed>0:
        conn = getLiveConnection()
    return conn

liveConnection = getLiveConnection()

def to_sleep(trading_date, sleep_params, minute,logger, sleep_value=5):
    trade_minute = pd.to_datetime(minute)
    cur_minute = pd.to_datetime(datetime.now())
    total_minute =  cur_minute - trade_minute
    total_minute = math.floor(total_minute.total_seconds() / 60)
    if pd.to_datetime(trading_date) < pd.to_datetime(datetime.now().date()):
        return
    
    if total_minute == 0:
        if sleep_params == 'missing_sleep':
            sleep_value = sleep_value
        elif sleep_params == 'minute_sleep':
            sleep_value  = abs((108)-datetime.now().time().second)
        elif sleep_params == 'first_minute_sleep':
            sleep_value = ((180+sleep_value)-datetime.now().time().second)
            logger.info(f"{sleep_params} for {sleep_value}")
        time.sleep(sleep_value)
        

def isHoliday(date):
    query = f'''select
                    (case
                        when count(*)= 1 then true
                        else false
                    end )
                from
                    tbl_nse_holidays
                where
                    dt_date = '{date}';'''
    try:
        conn = getLocalConnection()  
        data = pd.read_sql(query, conn)
        closeConnection(conn)
        return data.values[0][0]
    except Exception as e:
        logging.error(f"Error in checking holiday. Exception: {str(e)}")
        pass

def insertSentimentSignal(date,timestamp,j_sentiment,j_sentiment_updated,j_accurecy,result,db_action,logger):
    sentiment_list = json.dumps(j_sentiment)
    accurecy_list  = json.dumps(j_accurecy)
    updated_sentiment = json.dumps(j_sentiment_updated)

    if (db_action == 'insert'):
        try:
            # Define the PostgreSQL INSERT statement
            insert_query = """INSERT INTO public.tbl_news_signal_details_update
                (dt_date,j_sentiment,j_sentiment_updated, j_accuracy)
                VALUES(%s,%s,%s,%s);"""
            
            conn = getLocalConnection()
            curr = conn.cursor()
            curr.execute(insert_query, (date,sentiment_list,updated_sentiment,accurecy_list))
            conn.commit()
            conn.close()
            logger.info(f"""Model time => {timestamp} News Sentiment Signal inserted sucessfully with signal : {result} """)
        except Exception as e:
            logger.info(f"Error in Signal Insertion: {e}")
            pass
        db_action = 'update'
    else:  
        update_query = """UPDATE public.tbl_news_signal_details_update
                            SET j_sentiment=%s,j_sentiment_updated=%s, j_accuracy=%s
                            where dt_date = %s;"""
        try:
            conn = getLocalConnection()
            curr = conn.cursor()
            curr.execute(update_query, (sentiment_list,updated_sentiment,accurecy_list,date))
            conn.commit()
            conn.close()
            logger.info(f"Model time => {timestamp} News Sentiment Signal updated successfully with signal : {result}")
        except Exception as e:
            logger.info(f"Error in Signal Updation: {e}")
            pass
        db_action = 'update'
    return db_action

def InsertNewsSignal(cleaned_text,sent_,companies_list):
    try:
        conn = getLocalConnection()
        cursor = conn.cursor()
        insert_query = '''INSERT INTO tbl_news_sentiment_details_finbert (news, sentiment, symbol) VALUES (%s,%s,%s);'''
        cursor.execute(insert_query, (cleaned_text,sent_,companies_list))
        conn.commit()
        conn.close()
    except Exception as error:
        print("News Insertion error: ",error)
        pass

def getLiveNews(check_time,trading_time,logger):
    try:
        conn = getLocalConnection()
        select_query = f"""select time_stamp::varchar, news, sentiment, symbol 
                                       from tbl_news_sentiment_details_finbert 
                                       where time_stamp between '{check_time}' and '{trading_time}' 
                                       and length(news) > 50
                                       order by "time_stamp" asc;"""
        df_live_news   = pd.read_sql_query(select_query,conn) 
        if (len(df_live_news)> 0):
            return df_live_news
        else:
            return pd.DataFrame()
            
    except Exception as error:
        logger.info(f"Error in live news : {error}")
        pass

def getSymbols(trading_date):
    try:
        select_query = f"""select
                            distinct s_symbol
                        from
                            tbl_nseindexhistorical_2022
                        where
                            dt_date >=('{trading_date}'::date -interval '1 day' * 1)::date
                        union 
                        select
                            distinct s_symbol
                        from
                            tbl_nsecashhistorical_2022
                        where
                            dt_date >=('{trading_date}'::date -interval '1 day' * 1)::date
                        order by
                            s_symbol asc"""
        
        conn = getTickConnection()
        response = pd.read_sql(select_query,conn)
        return response
    except Exception as error:
        print(error)
        pass

def getSpotData(symbol,date,trading_time):
    try:
        if symbol.split('.')[-1] == 'NSE_IDX':
            minute_data = getNiftyData(date,trading_time)
        else:
            select_query = f"""select
                                s_symbol as "ASSET",
                                dt_date as "DATE",
                                dt_time::varchar(8) as "TIME",
                                n_open as "OPEN",
                                n_high as "HIGH",
                                n_low as "LOW",
                                n_close as "CLOSE",
                                n_volume as "VOLUME",
                                n_oi as "OI"
                            from
                                tbl_spot_data_day
                            where
                                dt_date = '{date}'
                                and s_symbol = '{symbol}'
                                and dt_time::time between ('{trading_time}' - interval '1 minute' * 30)::time and '{trading_time}'::time
                            order by dt_time asc;"""
        
            conn = getLocalConnection()
            i = 0
            while (i < 6):
                    minute_data = pd.read_sql(select_query,conn)
                    if len(minute_data) > 0:
                        if trading_time in minute_data['TIME'].values:
                            closeConnection(conn)
                            return minute_data
                    i = i+1
            closeConnection(conn)
        return minute_data
    except Exception as error:
        print(f"{symbol} spot data : {error}")
        pass

 

def getAllNiftyData(date,trading_time):
    try:
        select_query = f"""select
                            s_symbol as "ASSET",
                            dt_date::varchar as "DATE",
                            dt_time::varchar(8) as "TIME",
                            n_open as "OPEN",
                            n_high as "HIGH",
                            n_low as "LOW",
                            n_close as "CLOSE",
                            n_volume as "VOLUME",
                            n_oi as "OI"
                        from
                            tbl_nifty_minute
                        where
                            dt_date = '{date}'
                            and s_symbol = 'NIFTY 50.NSE_IDX'
                            and dt_time::varchar(5) <='{trading_time}'::varchar(5)
                            and dt_time::varchar(5) >='09:15:00'::varchar(5)
                        order by dt_time asc"""
        conn = getLocalConnection()
        response = pd.read_sql(select_query,conn)
        conn.close()
        return response
    except Exception as error:
        print(f"NIFTY spot data :",error)
        pass
    
    
def getNiftyData(date, trading_time):
    tickConn = reconnect(liveConnection)
    try:
        select_query = f'''select 
                        (dt_date::date || ' ' || dt_time::time)::timestamp as "DATETIME",
                        n_ltp 
                    from 
                        tbl_nse_second_live 
                    where
                        s_symbol::varchar = 'NIFTY 50.NSE_IDX'
                        and dt_date::date = '{date}'
                        and dt_time >= '{trading_time}'::time-interval '30 minutes'
                        and dt_time <= '{trading_time}';'''
                        
        response = pd.read_sql(select_query, tickConn)
        response = response.set_index('DATETIME')['n_ltp'].resample('1Min').ohlc().reset_index()
        response.columns = [cl.upper() for cl in response.columns]
        response['DATE'] = [str(x)[:10] for x in response['DATETIME']]
        response['TIME'] = [str(x)[11:17]+'45' for x in response['DATETIME']]
        return response
    except Exception as error:
        logging.info(f"NIFTY spot data :{error}")
        pass

                                     

def insertUpdateFunction(dbAction, details,logger, minute, sentimentValue):
    if dbAction == 'preMarketUpdate':
        try:
            query = '''update
                            tbl_news_sentiment_signal_details
                        set
                            n_premarket_sentiment = %s
                        where
                            dt_date = %s;'''
            
            value = (details["preMarketSentiment"], details["Date"])
            db_connection = getLocalConnection()
            cursor = db_connection.cursor()
            cursor.execute(query, value)
            db_connection.commit()
            closeConnection(db_connection)

            logger.info(f"Model Date & Time => {details['Date']} {minute}, Pre-market sentiment : {sentimentValue} inserted successfuly.")
        except Exception as err:
            logger.info(f"Model Date & Time => {details['Date']} {minute}, Not able to insert Pre-market sentiment. =>{str(err)}")
            pass

    elif dbAction == 'preMarketInsert':
        try:
            query = '''insert into
                        tbl_news_sentiment_signal_details (dt_date,
                        n_preMarket_sentiment)
                        values(%s,%s);'''
            
            values = (details["Date"], details["preMarketSentiment"])
            db_connection = getLocalConnection()
            cursor = db_connection.cursor()
            cursor.execute(query, values)
            db_connection.commit()
            closeConnection(db_connection)

            logger.info(f"Model Date & Time => {details['Date']} {minute}, Pre-market sentiment : {sentimentValue} inserted successfuly.")
        except Exception as err:
            logger.info(f"Model Date & Time => {details['Date']} {minute}, Not able to insert Pre-market sentiment. =>{str(err)}")
            pass

    return dbAction

def isPresent(date):
    query = f'''select (case when count(*) = 1 then true
                        else false end)
                    from
                        tbl_news_sentiment_signal_details
                    where
                        dt_date = '{date}';'''
    try:
        db_connection = getLocalConnection()
        data = pd.read_sql(query, db_connection)
        closeConnection(db_connection)
        return data.values[0][0]
    except Exception as e:
        logging.error(f"Error in isPresent function. =>{str(e)}")



# def updateMinuteAccurecy(date,j_accurecy,time,logger):
#     accurecy_list = json.dumps(j_accurecy)
#     update_query = """UPDATE public.tbl_news_sentiment_signal_details
#                         SET j_accuracy=%s where dt_date = %s;"""
    
#     try:
#         conn = getLocalConnection()
#         curr = conn.cursor()
#         curr.execute(update_query, (accurecy_list,date))
#         conn.commit()
#         conn.close()
#         logger.info(f"Model time => {time} Minute accurecy update successfully")
#     except Exception as e:
#         logger.info(f"Model time => {time} Minute accurecy error: {e}")
#         pass


def updateMinuteAccurecy(date, j_accurecy, j_pl, time, logger):
    accurecy_list = json.dumps(j_accurecy)
    pl_list = json.dumps(j_pl)
   
    update_query = """UPDATE public.tbl_news_sentiment_signal_details
                        SET j_accuracy=%s,j_pl=%s where dt_date = %s;"""
    try:
        conn = getLocalConnection()
        curr = conn.cursor()
        curr.execute(update_query, (accurecy_list, pl_list, date))
        conn.commit()
        conn.close()
        logger.info(f"Model time => {time} Minute accurecy update successfully")
    except Exception as e:
        logger.info(f"Model time => {time} Minute accurecy error: {e}")
        pass
  
def niftyDataFunction(date,symbol,minute,j_ohlc,ohlc_action,logger):
    j_ohlc = json.dumps(j_ohlc)
    
    if ohlc_action == 'insert':
        insert_query = f"""INSERT INTO public.tbl_nifty50_details (dt_date, s_symbol, j_ohlc) VALUES('{date}', '{symbol}','{j_ohlc}'::jsonb);"""
        try:
            db_connection = getLocalConnection()
            cursor = db_connection.cursor()
            cursor.execute(insert_query)
            db_connection.commit()
            closeConnection(db_connection)
            logger.info(f"Model Time => {minute}, Nifty data inserted Successfully.")
  
        except Exception as err:
            logger.info(f"Model Time => {minute}, Not able to insert :{str(err)}")  
            pass
    else:
        update_query = f"""UPDATE public.tbl_nifty50_details
                            SET j_ohlc='{j_ohlc}'::jsonb WHERE dt_date='{date}';"""
        
        try:
            db_conn = getLocalConnection()
            cursor = db_conn.cursor()
            cursor.execute(update_query)
            db_conn.commit()
            closeConnection(db_conn)
            logger.info(f"Model Time => {minute}, Nifty data updated Successfully.")
  
        except Exception as err:
            logger.info(f"Model Time => {minute}, Not able to update :{str(err)}")  
            pass
    return 'update'

def dictToDataframe(date,j_sentiment):
    df1 = [{"timestamp":pd.to_datetime(date+" "+key),"result": value} for key, value in j_sentiment.items()]
    signal_dataframe = pd.DataFrame.from_dict(df1)
    return signal_dataframe

def updateAccurecy(n_news_accurecy,n_eco_accurecy,date,trading_time,logger):
    update_query = """UPDATE public.tbl_news_sentiment_signal_details
                        SET n_news_accurecy=%s,n_eco_accurecy=%s  where dt_date = %s;"""
    
    try:
        conn = getLocalConnection()
        curr = conn.cursor()
        curr.execute(update_query,(n_news_accurecy,n_eco_accurecy,date))
        conn.commit()
        conn.close()
        logger.info(f"Model time => {trading_time} news signal accurecy:{n_news_accurecy}, economic accurecy: {n_eco_accurecy} update successfully")
    except Exception as e:
        logger.info(f"Error in updation news signal accurecy:{n_news_accurecy}, economic accurecy: {n_eco_accurecy} : {e}")
        pass

def getPremarketSentiment(date):
    eco_sentiment = f"""select
                    dt_date::varchar,
                    round(n_sentiment_score::numeric, 4) as n_sentiment_score
                from
                    tbl_sentiment_score
                where
                    dt_date = '{date}'"""
    
    news_sentiment = f"""SELECT n_premarket_sentiment
                            FROM tbl_news_sentiment_signal_details
                            WHERE dt_date='{date}'"""
    
    try:
        conn = getLocalConnection()
        eco_response = pd.read_sql(eco_sentiment,conn)
        news_response = pd.read_sql(news_sentiment,conn)
        return eco_response.iloc[0]["n_sentiment_score"],news_response.iloc[0]["n_premarket_sentiment"]
    except Exception as error:
        print(f"pre_market sentiment: {error}")
        pass


def getNiftySpotData(date, trading_time):
    tickConn = reconnect(tickConnection)
    try:
        select_query = f'''select 
                        (dt_date::date || ' ' || dt_time::time)::timestamp as "DATETIME",
                        n_ltp 
                    from 
                        tbl_nse_second_live 
                    where
                        s_symbol::varchar = 'NIFTY 50.NSE_IDX'
                        and dt_date::date = '{date}'
                        and dt_time >= '{trading_time}'
                        and dt_time::varchar <= ('{trading_time}'::varchar(5)||'59');'''
        response = pd.read_sql(select_query, tickConn)
        response = response.set_index('DATETIME')['n_ltp'].resample('1Min').ohlc().reset_index()
        response.columns = [cl.upper() for cl in response.columns]
        response['DATE'] = [str(x)[:10] for x in response['DATETIME']]
        response['TIME'] = [str(x)[11:19] for x in response['DATETIME']]
        return response
    except Exception as error:
        logging.info(f"NIFTY spot data :{error}")
        pass
    

