import numpy as np
import pandas as pd
from utils import marketTimeFilter
from sentiment_model import sentimentAndWeightGenerator
import warnings
warnings.filterwarnings("ignore")

import nltk
import math
from statistics import mean
from datetime import date, timedelta
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from variables import config_params as config



def filter_news(df, news_column_name):  
    # Dropping NA values  
    df = df.dropna()  
    # # dropping duplicate values  
    df = df.drop_duplicates()    
    # # Reading the keywords from the excel file  
    keywords = [keyword.lower() for keyword in pd.read_excel(config['weight_data_path'],sheet_name = config['filtering_keywords'])['keywords'].to_list()]    
    # # Function that returns True if keyword exist inside a text  
    
    def contains_keyword(text):    
        text_lower = text.lower()   
        return any(keyword in text_lower for keyword in keywords)  
    # # Applying a function and creating a mask that tells which news we should take  
    
    mask = df[news_column_name].apply(contains_keyword)  
    # # Creating the final filtered dataframe  
    filtered_df = df[mask]  
    return filtered_df.reset_index(drop = True)


def scaling_function(x:float)-> float:
    """This function scales the news sentiments in between 1 and -1

    Args:
        x (float): mean sentiment of the pre-market news sentiments

    Returns:
        float: scaled data
    """
    x_ = ((2*(x - config['min'])/(config['max'] - config['min'])) -1)
    
    if x_ > 1:
        return math.floor(x_)
    elif x_ < -1:
        return math.ceil(x_)
    else:
        return x_


def preMarketSentimentGeneration(df: pd.DataFrame,
                                    news_column:str,
                                    date_column:str,
                                    )->float:
    """_summary_
    Args:
        df_news (pd.DataFrame): Pre market news dataframe
        news_column (str): News Summary column of the premarket news dataset
        date_column (str): Datetime index column of the premarket news dataset
    Returns:
        float: Premarket Sentiment Score
    """
    # Filtering the news
    df_news = filter_news(df, news_column)
    appended_data = []
    # print("df_news",df_news)
    for i in range(len(df_news)):
        if marketTimeFilter(str(df_news.iloc[i][date_column])) == 'no':
            news = df_news.iloc[i][news_column]
            timestamp = df_news.iloc[i][date_column]
            odf = sentimentAndWeightGenerator(str(news),str(timestamp))
            appended_data.append(odf)
        else:
            pass
    output_df = pd.concat(appended_data)
    unscaled_sent = output_df['Sentiment_Score'].mean()
    scaled_sent = scaling_function(unscaled_sent)
    return round(scaled_sent,4)

