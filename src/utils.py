# Importing Necessary libraries
import re
import sys
import numpy as np
import pandas as pd
from statistics import mode
from datetime import datetime, timedelta
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.util import ngrams
from functools import partial
# from config import weight_data_path, nifty_companies
# Downloading necessary files for nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import torch
import re
from transformers import AutoTokenizer, BertModel, BertTokenizer
from torch.utils.data import DataLoader
from torch import nn, optim

# To remove the BERT model warning
from transformers import logging
logging.set_verbosity_error()

from variables import config_params as config

# Function to load yaml configuration file
# def load_config(config_name):
#     with open(os.path.join(CONFIG_PATH, config_name)) as file:
#         config = yaml.safe_load(file)

#     return config


# config = load_config("config.yaml")

# Decontracting strings
def decontract_strings(string: str)-> str:
    """Decontracting strings.
    
    :param num: Input str
    """ 
    
    # Doing for ' symbol
    # specific
    string = re.sub(r"won't", "will not", string)
    string = re.sub(r"can\'t", "can not", string)

    # general
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'t", " not", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'m", " am", string)
    
    # Doing similar for ’ symbol
    string = re.sub(r"won’t", "will not", string)
    string = re.sub(r"can\’t", "can not", string)

    # general
    string = re.sub(r"n\’t", " not", string)
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\’s", " is", string)
    string = re.sub(r"\’d", " would", string)
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\’m", " am", string)
    return string



# Emoji remover
def remove_emoji(string: str)-> str:
    """Cleanes emoji inside text.
    
    :param num: Input str
    """
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(string))



# Removing Stop words

# List of words we dont want to remove from the list of stop words
list_of_words = config['stopwords_to_ignore']

# Creating a custom list of stopwords
stopwords_ = set(stopwords.words('english'))

s = set(list_of_words)
custom_stopwords_ = list(stopwords_ - s)


def remove_sw(text: str)-> str:
    """Removes the stopwords.
    
    :param num: Input str
    """
    words = [word for word in text.split() if word.lower() not in custom_stopwords_]
    return " ".join(words)




# Final Cleaning
def clean_text(text: str)-> str:
    """Cleanes every unnecessary special characters, spaces, tabs etc. inside text and turns the text into lowercase.
    
    :param num: Input str
    """
    text= re.sub("'", "",text)
    text= re.sub('@[A-Za-z0-9]+','',text ) #removing mentions
    text= re.sub(r"[^a-zA-Z0-9]", " ", text) # Removing non letters
    text= re.sub("#",'',text) #removing #
    text= re.sub('RT[\s]+',' ',text) # removing Retexts
    text= re.sub(r"http\S+", "",text) #removing links
    text= re.sub("[+%|):@(€¥£—•;=*]"," ",text) # Removing some special symbols
    text= re.sub(r'[^\x00-\x7F]+',' ', text) # Removinig emojies
    text= re.sub('^\d+\s|\s\d+\s|\s\d+$',' ', text) 
    text= re.sub('’','', text) # Remoing special character
    text= re.sub('₹',' ', text) # Removing special character
    text= re.sub('“','', text) # Removing special  character
    text= re.sub('—',' ', text) # Removing special character
    text= re.sub('—',' ', text) # Removing special character
    text= re.sub('”','', text) # Removing special character
    text= re.sub('»','', text) # Removing special character
    text= re.sub('“','', text) # Removing special character
    text= re.sub(r'[0-9]', '', text)
    text= re.sub(' +', ' ',text) # Removing extra spaces and tabs
    text= text.lower() # Transforming text into lower
    return text


# Lematization
lemmatizer = WordNetLemmatizer()

def lemmatize(text: str)-> str:
    """Lemmatize the input texts.
    
    :param num: Input str
    """
    words = text.split()
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    return ' '.join(words)


########################################################################################
#                                   Some Basic Functions                               #
########################################################################################

def slice_news(text:str)->str:
    """
    This function takes a news as input and if the number of sentences inside the news is greater than 3 it will slice it into three such sub sentences. 
    If not return the previous sentence.
    Args:
        text (str): News 
    Returns:
        (list): list_of_trigram_sentences
    """
    # Toekni
    sentences = nltk.sent_tokenize(text)
    if len(sentences)<= 3:
        return list(sentences)
    else:
        output = []
        for i in range(0,len(sentences),3):
            output.append(' '.join(sentences[i:i+3]))
        return output

def list_text_cleaning(list_of_sentences:list)-> list:
    """
    This function cleans list of sentences 
    """
    output = []
    for sentences_ in list_of_sentences:
        #print(sentences_)
        processed_sentence = remove_emoji(sentences_)
        processed_sentence = decontract_strings(processed_sentence)
        processed_sentence = clean_text(processed_sentence)
        processed_sentence = lemmatize(processed_sentence)
        output.append(processed_sentence)
        
    return output

########################################################################################
#                   Functions to Generate Weights from the News                        #
########################################################################################

# Initializing weight dataframes
nifty_50_comapny_weights_df = pd.read_excel(config['weight_data_path'],
                                  sheet_name = config['company_sheet'] )
    
nifty50_sector_weight_df    = pd.read_excel(config['weight_data_path'],
                                  sheet_name = config['sector_sheet'] )

pos_word_weight_df          = pd.read_excel(config['weight_data_path'],
                                  sheet_name = config['positive_word_sheet'] )

neg_word_weight_df          = pd.read_excel(config['weight_data_path'],
                                  sheet_name = config['negative_word_sheet'] )

pos_bigram_weight_df        = pd.read_excel(config['weight_data_path'],
                                  sheet_name = config['pos_bigram_sheet'] )

neg_bigram_weight_df        = pd.read_excel(config['weight_data_path'],
                                  sheet_name = config['neg_bigram_sheet'] )

keyword_weight_df           = pd.read_excel(config['weight_data_path'],
                                  sheet_name = config['keyword_sheet'] )



# Weights from company and industry
def company_industry_weight_generator(news:str,
                                      sentiment:int)->float:
    """ This function takes one news/tweet as input and generate weight based on company and industry weights in NIFTY50.

    Args:
        news (str): news/tweet in string format
        sentiment (int): -1: Negative ; 1: Positive & 0: Neutral

    Returns:
        average company weights (float), average industry weight (float)
    """
    matching_nifty_50_companies_weights  = []
    matching_nifty_50_industries_weights = []
    #==================    Checking if a company name in nifty50 company dataset exists inside the news    ==================#
    for company in nifty_50_comapny_weights_df['NAME']:
      if re.search(r'\b' + re.escape(company) + r'\b', news, re.IGNORECASE):
          #print(company)
          matching_nifty_50_companies_weights.append(nifty_50_comapny_weights_df[nifty_50_comapny_weights_df['NAME'] == company]['ASSIGNED_WEIGHT'].values)
          industry = nifty_50_comapny_weights_df[nifty_50_comapny_weights_df['NAME'] == company]['SECTOR'].values
          #print(industry[0])
          matching_nifty_50_industries_weights.append(nifty50_sector_weight_df[nifty50_sector_weight_df['SECTOR'] == str(industry[0])]['ASSIGNED_WEIGHT'].values)
   
    # print("Industry weights: ", matching_nifty_50_industries_weights)
    # print("Comapany weights: ", matching_nifty_50_companies_weights)
    # print("Industry mean weights: ", np.mean(matching_nifty_50_industries_weights))
    # print("Comapany mean weights: ", np.mean(matching_nifty_50_companies_weights))
    return np.nanmean(matching_nifty_50_companies_weights) * sentiment, np.nanmean(matching_nifty_50_industries_weights) * sentiment


# Weights from keywords
def keyword_weight_generator(news:str,
                             sentiment:int)->float:
    """ This function takes one news/tweet as input and generate weight based on selected keywords.

    Args:
        news (str): news/tweet in string format
        sentiment (int): -1: Negative ; 1: Positive & 0: Neutral

    Returns:
        keyword weight (float) 
        
    """
    keyword_weights  = []

    #==================    Checking if a company name in nifty50 company dataset exists inside the news    ==================#
    for keyword in keyword_weight_df['KEYWORD']:
      if re.search(r'\b' + re.escape(keyword) + r'\b', news, re.IGNORECASE):
        #   print(keyword)
          keyword_weights.append(keyword_weight_df[keyword_weight_df['KEYWORD'] == keyword]['ASSIGNED_WEIGHT'].values)
          #print(industry[0])
   
    # print("Keyword weights: ", keyword_weights)
    # print("Keyword mean weights: ", np.nanmean(keyword_weights))
    return np.nanmean(keyword_weights) * sentiment


# Weights from words
def unique_word_weight_generator(news:str,
                                 sentiment:int)->float:
    """This function take one news/tweet as input and weight it accroding to the words.

    Args:
        news (str): news or tweet in string format
        sentiment (int): -1: Negative ; 1: Positive & 0: Neutral

    Returns:
        float: sentiment weight based on the news weight (it can be both postitive and negative)
    """

    

    #==================    Checking if a company name in nifty50 company dataset exists inside the news    ==================#
    keyword_weights  = []
    
    if sentiment == -1:
        
        for word in neg_word_weight_df['Word']:
            if re.search(r'\b' + re.escape(word) + r'\b', news, re.IGNORECASE):
                # print(word)
                keyword_weights.append(neg_word_weight_df[neg_word_weight_df['Word'] == word]['Weights'].values)
                
        # print(keyword_weights)
        return np.nanmean(keyword_weights) * -1
    
    elif sentiment == 1:
        
        for word in pos_word_weight_df['Word']:
            if re.search(r'\b' + re.escape(word) + r'\b', news, re.IGNORECASE):
                # print(word)
                keyword_weights.append(pos_word_weight_df[pos_word_weight_df['Word'] == word]['Weights'].values)
        # print(keyword_weights)
        return np.nanmean(keyword_weights)
    
    else:
        pos_word_weight = []
        neg_word_weight = []
        for word in pos_word_weight_df['Word']:
            if re.search(r'\b' + re.escape(word) + r'\b', news, re.IGNORECASE):
                # print(word)
                pos_word_weight.append(pos_word_weight_df[pos_word_weight_df['Word'] == word]['Weights'].values)
        mean_pos_word_weight = np.nanmean(pos_word_weight)
        
        for word in neg_word_weight_df['Word']:
            if re.search(r'\b' + re.escape(word) + r'\b', news, re.IGNORECASE):
                # print(word)
                neg_word_weight.append(neg_word_weight_df[neg_word_weight_df['Word'] == word]['Weights'].values)
        mean_neg_word_weight = np.nanmean(neg_word_weight)
        
        over_all_weight = mean_pos_word_weight - mean_neg_word_weight
        if over_all_weight > 0.4:
            return over_all_weight
        elif over_all_weight < -0.1:
            return mean_neg_word_weight * -1
        else:
            return 0


# Weights from bigrams
def unique_bigram_weight_generator(news:str, 
                                   sentiment:int)->float:
    """This function take one news/tweet as input and weight it accroding to the words.

    Args:
        news (str): news or tweet in lemmatized format
        sentiment (int): -1: Negative ; 1: Positive & 0: Neutral

    Returns:
        float: sentiment weight based on the mentioned bigrams (it can be both postitive and negative)
    """

    

    #==================    Checking if a company name in nifty50 company dataset exists inside the news    ==================#
    keyword_weights  = []
    
    if sentiment == -1:
        
        for word in neg_bigram_weight_df['ngrams']:
            # print("Searching for negative bi gram inside the news", word)
            if re.search(r'\b' + re.escape(word) + r'\b', news, re.IGNORECASE):
                # print("Negative Bigram found: ", word)
                keyword_weights.append(neg_bigram_weight_df[neg_bigram_weight_df['ngrams'] == word]['weights'].values)
                
        # print(keyword_weights)
        return np.nanmean(keyword_weights) * sentiment
    
    elif sentiment == 1:
        
        for word in pos_bigram_weight_df['ngrams']:
            # print("Searching for positive bi gram inside the news", word)
            if re.search(word, news):
                # print("Positive Bi gram found: ",word)
                keyword_weights.append(pos_bigram_weight_df[pos_bigram_weight_df['ngrams'] == word]['weights'].values)
        # print(keyword_weights)
        return np.nanmean(keyword_weights) * sentiment
    
    else:
        pos_bigram_weight = []
        neg_bigram_weight = []
        for bigram in pos_bigram_weight_df['ngrams']:
            # print("Searching for positive bi gram inside the news",bigram)
            if re.search(r'\b' + re.escape(bigram) + r'\b', news, re.IGNORECASE):
                # print("Positive Bi gram found: ",bigram)
                pos_bigram_weight.append(pos_bigram_weight_df[pos_bigram_weight_df['ngrams'] == bigram]['weights'].values)
        mean_pos_bigram_weight = np.nanmean(pos_bigram_weight)
        
        for bigram in neg_bigram_weight_df['ngrams']:
            # print("Searching for negative bi gram inside the news",bigram)
            if re.search(r'\b' + re.escape(bigram) + r'\b', news, re.IGNORECASE):
                # print("Negative Bi gram found: ",bigram)
                neg_bigram_weight.append(neg_bigram_weight_df[neg_bigram_weight_df['ngrams'] == bigram]['weights'].values)
        mean_neg_bigram_weight = np.nanmean(neg_bigram_weight)
        
        over_all_weight = mean_pos_bigram_weight - mean_neg_bigram_weight
        if over_all_weight > 0.4:
            return over_all_weight
        elif over_all_weight < -0.1:
            return mean_neg_bigram_weight * -1
        else:
            return 0



########################################################################################
#                                   Helper Function                                    #
########################################################################################


def marketTimeFilter(dt = "Timestamp")->str:
    """ This function take timestamp of news in YYYY-MM-DD HH:MM:SS formart and return yes if the news came during market hour else return no.

    Args:
        dt (str, Defaults to "Timestamp"): Timestamp of the news in YYYY-MM-DD HH:MM:SS

    Returns:
        str: "yes" or "no"
    """
    time_hh = int(dt[11:13])
    time_mm = int(dt[14:16])
    if (time_hh >= 10) & (time_hh <= 14):
        return "yes"
    
    elif (time_hh ==9) & (time_mm >=15):
        return 'yes'
    
    elif (time_hh ==15) & (time_mm <=30):
        return 'yes'
    
    else:
        return "no"

    

# Creating a function that slices the news dataframe accroding to the current time and lookback minute    
def df_slicer(df:str,
              current_time:str,
              lookback_time:str,
              time_index_column:str)->pd.DataFrame:
    """This function slice the news dataframe accroding to the current time and lookback time. 

    Args:
        df (str): News Datafame
        current_time (str): Current time in HH:MM:SS format
        lookback_time (str): Look back time in HH:MM:SS format
        time_index_column (str): Time index column

    Returns:
        pd.DataFrame: Sliced dataframe
    """
    
    df[time_index_column] = pd.to_datetime(df[time_index_column])
    
    df = df.set_index(df[time_index_column])
    
    df = df[(df[time_index_column] >= lookback_time) &
            (df[time_index_column] <= current_time)]
    
    df = df.sort_index()
    df = df.reset_index(drop = True)
    df[time_index_column] = df[time_index_column].astype(str).apply(lambda x: x[11:])
    
    return df
    
    
 
 
 
# Calculate weighted average mean
def sentiment_weighted_average(df:pd.DataFrame,
                               sentiment_column:str)->float:
    """This function creates a weighted mean of sentiment score from the input data.
       If the news arrives recently it gives more weight to it.
    Args:
        df (pd.DataFrame): News Sentiment dataframe
        sentiment_column (str): Name of the Sentiment Score Column
    Returns:
        float: Weighted mean of sentiment score for the whole dataframe
    """
    n = len(df)
    total_n = n*(n-1)/2
    #weighted_sum = 0
    list_of_weighted_sentiment = []
    
    for i in range(len(df)):
        weight = (i+1)/n
        weighted_sentiment = weight * df.iloc[i][sentiment_column]
        list_of_weighted_sentiment.append(weighted_sentiment)
        #weighted_sum += weighted_sentiment

    weighted_avg = np.nanmean(list_of_weighted_sentiment)
    return weighted_avg