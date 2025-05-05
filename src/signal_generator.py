# from data_collector import *
import pandas as pd

def signalGenerator(correlation_dataframe:pd.DataFrame,
                     opportunity_dataframe:pd.DataFrame,
                     news_dataframe:pd.DataFrame)->dict:
    """This function generates signals (1 => bullish signal & 
                                       -1 => bearish signal)
    Args:
        correlation_dataframe (pd.DataFrame): _description_
        opportunity_dataframe (pd.DataFrame): _description_
        news_dataframe (pd.DataFrame): _description_
    Returns:
        dict: _description_
    """
    
    try:
        last_instance_opportunity_dataframe = opportunity_dataframe.tail(n=1).reset_index(drop= True)
        
        # Bullish Signal Generating
        if (news_dataframe[news_dataframe['symbol'].astype(str) != '[]']['sentiment'].astype(int).mean() > news_dataframe['sentiment'].astype(int).mean()) & (abs(correlation_dataframe[correlation_dataframe['ASSET'].isin([symbol for sublist in news_dataframe['symbol'].values for symbol in sublist])]['Corr'].mean()) > 0.6) & (last_instance_opportunity_dataframe['opportunity_1'][0] == 1):
            return 1
        elif (news_dataframe[news_dataframe['symbol'].astype(str) != '[]']['sentiment'].astype(int).mean() < news_dataframe['sentiment'].astype(int).mean()) & (abs(correlation_dataframe[correlation_dataframe['ASSET'].isin([symbol for sublist in news_dataframe['symbol'].values for symbol in sublist])]['Corr'].mean()) > 0.6) & (last_instance_opportunity_dataframe['opportunity_2'][0] == 1):
            return -1
        elif  (str([symbol for sublist in news_dataframe['symbol'][-180:].values for symbol in sublist]) != '[]')  & (news_dataframe['sentiment'].astype(int)[-10:].mean() > news_dataframe['sentiment'].astype(int).mean())  & (last_instance_opportunity_dataframe['opportunity_1'][0] == 1):
            return 1
        # Bearish Signal Generation
        elif (str([symbol for sublist in news_dataframe['symbol'][-180:].values for symbol in sublist]) != '[]')  & (news_dataframe['sentiment'].astype(int)[-10:].mean() < news_dataframe['sentiment'].astype(int).median()) & (last_instance_opportunity_dataframe['opportunity_2'][0] == 1):
            return -1

        else:
            return 'NA'
    except:
        return 'NA'

    
