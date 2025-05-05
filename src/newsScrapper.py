import pandas as pd
from datetime import datetime, timedelta 
import urllib.request, json
import logging
from config import isHoliday
# pd.set_option('display.max_rows',30000)



def change(x)->str:
    x=str(x)
    x = x.split('+')[0].replace('T',' ')
    return x


def parse_datetime(x: str)-> str:
    return pd.to_datetime(x).strftime("%Y-%m-%dT%H:%M:%S+05:30")


def getNewsTweetsForADate(date, allNewsDF):
    latestNewsDF = pd.DataFrame()
    newsDf = pd.DataFrame()
    date_time = str(f'{date} 09:12:00')
    current_date_time = parse_datetime(date_time)
    # current_date_time2 = parse_datetime(f'{date} 15:30:00')
    try:
        newsURL = f"https://api.vistaintelligence.ai/requesthandler/v1/news/getNewsTweetsForADate?date={str(date)}&type=news"
        with urllib.request.urlopen(newsURL) as res:
            data = res.read()
            newsDf = pd.DataFrame(json.loads(data), columns=['datePublished', 'summary', 'category'])
    except Exception as err:
        logging.info(f"ERROR WHEN TRY TO GET NEWS PER MINUTE USING HTTPS API CALL: {str(err)}")
        pass
    try:
        newsDf = newsDf[newsDf['category'] != 'Other' ]
        newsDf = newsDf.sort_values(by = 'datePublished', ascending = False)
        newsDf = newsDf[newsDf['datePublished']>= current_date_time]
        # newsDf = newsDf[newsDf["datePublished"] <= current_date_time2]
        newsDf.rename(columns = {'datePublished':'Timestamp'}, inplace = True)
        newsDf["DATE"] = pd.to_datetime(newsDf["Timestamp"]).dt.strftime("%Y-%m-%d")
        newsDf["TIME"] = pd.to_datetime(newsDf["Timestamp"]).dt.strftime("%H:%M:%S")
        newsDf["Timestamp"] = newsDf["Timestamp"].apply(lambda x: change(x))
        if(len(allNewsDF) != len(newsDf)):
            latestNewsDF = pd.concat([allNewsDF, newsDf], ignore_index=True).drop_duplicates(keep=False, subset=['Timestamp'])
            allNewsDF = newsDf
    except:
        pass
    return allNewsDF, latestNewsDF


def dateGenerator(today, lastDate, sunday, saturday, friday):
    lastDate = str((pd.to_datetime(today) - timedelta(days=1)).date())
    if isHoliday(lastDate) == False:
        if pd.to_datetime(today).weekday() == 0:
            friday = str((pd.to_datetime(today) - timedelta(days=3)).date())
            saturday = str((pd.to_datetime(today) - timedelta(days=2)).date())
            sunday = str((pd.to_datetime(today) - timedelta(days=1)).date())
            lastDate = ''  
    else:
        friday = str((pd.to_datetime(today) - timedelta(days=4)).date())
        saturday = str((pd.to_datetime(today) - timedelta(days=3)).date())
        sunday = str((pd.to_datetime(today) - timedelta(days=2)).date())
    return lastDate, sunday, saturday, friday
        

def fetchNewsForPreMarket(date):
    """It return the today's and previous day news dataframe according to dates.
    Args:
        date (string): YYYY-MM-DD
    Returns:
        It return the the whole news data frame.
    """
    today = str(date)
    friday=''
    saturday=''
    sunday = ''
    lastDate = ''
    df = pd.DataFrame()
    lastDate, sunday, saturday, friday = dateGenerator(today, lastDate, sunday, saturday, friday)
    # print("dates",lastDate, sunday, saturday, friday)
    def fetch_news(date, day=None, time_condition=None):
        try:
            newsURl = f"https://api.vistaintelligence.ai/requesthandler/v1/news/getNewsTweetsForADate?date={date}&type=news"
            res = urllib.request.urlopen(newsURl)
            data = res.read()
            news_df = pd.DataFrame(json.loads(data), columns=['datePublished', 'summary', 'category'])
            news_df = news_df[news_df['category'] != 'Other']
            news_df = news_df.sort_values(by='datePublished', ascending=False)
            if time_condition:
                if day == 'today':
                    news_df = news_df[time_condition >= news_df["datePublished"]]
                else:
                    news_df = news_df[time_condition <= news_df["datePublished"]]
            return news_df
        except Exception as err:
            logging.error(f"Error in fetching pre-market news for {date}, {err}")
    
    try:
        today_df = fetch_news(today, day='today', time_condition=f'{today}T09:00:00+05:30')
        df = pd.concat([df, today_df], ignore_index=True)
    except:
        pass

    try:
        last_date_df = fetch_news(lastDate, time_condition=f'{lastDate}T15:30:00+05:30')
        df = pd.concat([df, last_date_df], ignore_index=True)
    except:
        pass

    try:
        sunday_df = fetch_news(sunday)
        df = pd.concat([df, sunday_df], ignore_index=True)
    except:
        pass

    try:
        saturday_df = fetch_news(saturday, time_condition=f'{saturday}T15:30:00+05:30')
        df = pd.concat([df, saturday_df], ignore_index=True)
    except:
        pass

    try:
        friday_df = fetch_news(friday, time_condition=f'{friday}T15:30:00+05:30')
        df = pd.concat([df, friday_df], ignore_index=True)
    except:
        pass

    return df
   