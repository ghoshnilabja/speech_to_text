# Importing the necessary libraries
import pandas as pd
import requests
import sqlite3

# Utility Functions
def position_finder(_list_, threshold):
    lngth = int(len(_list_))
    list_of_positions = []
    if threshold>= 0:
        for i in range(0,lngth):
            if _list_[i] >= threshold:
                list_of_positions.append(i)
    else:
        for i in range(0,lngth):
            if _list_[i] <= threshold:
                list_of_positions.append(i)
    if len(list_of_positions) != 0:
        return list_of_positions[0]
    else:
        return 999999
    
    

def preMarketSentChecker(date_: str,
                            sent_score:float,
                            market_data:pd.DataFrame,
                            point_movement = 6,
                            time_to_check = 20,
                            threshold = 0)-> int:
    """This Function returns 1 if sentiment score is correct

    Args:
        date_ (str): Trading day in yyyy-mm-dd format
        sent_score (float): Sentiment Score
        market_data (pd.DataFrame): Nifty 50 futures data till 9:36 am
        point_movement (int, optional): Price movement we want to check. Defaults to 6.
        time_to_check (int, optional): Timeframe we want to check. Defaults to 20.
        threshold (int, optional): Sentiment Threshold. Defaults to 0.

    Returns:
        int:  1 -> correct/ 0 -> false
    """

    # Initializing the Nifty futures dataframe column names
    date_col = 'DATE'
    open_col = 'OPEN'
    high_col = 'HIGH'
    low_col  = 'LOW'
    close_col= 'CLOSE'
    
    # Getting the opening price
    opening_price = market_data[market_data[date_col] == date_][open_col].iloc[1]
    closing_prices = market_data[market_data[date_col] == date_][close_col][2:time_to_check].tolist() - opening_price
    high_prices  = market_data[market_data[date_col] == date_][high_col][2:time_to_check].tolist() - opening_price
    low_prices   = market_data[market_data[date_col] == date_][low_col][2:time_to_check].tolist() - opening_price

    # Finding the prices positions inside the list
    low_price_occuring_position = position_finder(low_prices, -1 * point_movement)
    close_price_occuring_position = position_finder(closing_prices, point_movement)
    close_price_occuring_position_neg = position_finder(closing_prices, -1 * point_movement)
    high_price_occuring_position = position_finder(high_prices, point_movement)

    if (sent_score <= threshold) and (low_price_occuring_position <= close_price_occuring_position):
        return 1
    elif (sent_score > threshold) and (high_price_occuring_position <= close_price_occuring_position_neg):
        return 1
    else:
        return 0



