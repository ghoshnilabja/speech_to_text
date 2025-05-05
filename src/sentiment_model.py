# Loading config files
import sys
# from setup import *
from variables import config_params as config, model_parameters as parms

sys.path.append(config['utils_file_path'])
from utils import *

# Importing Libraries
import os
import re
import sys
import numpy as np
import pandas as pd
from statistics import mode
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.util import ngrams
from functools import partial
import torch
import re
from transformers import AutoTokenizer, BertModel, BertTokenizer
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import torch.nn as nn
import torch.optim as optim
# To remove the BERT model warning
from transformers import logging
logging.set_verbosity_error()

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Device :", device)
# Load the pre-trained BERT model and tokenizer
model_name = config['FinBERT_Tone_Model']
tokenizer = BertTokenizer.from_pretrained(config['FinBERT_Tone_Model_Path'])
bert_model = BertModel.from_pretrained(config['FinBERT_Tone_Model_Path']).to(device)



# Defining classifier model class
class CNN_DNN_Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_DNN_Classifier, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels    = parms['input_channels'], 
                      out_channels   = parms['output_channels'], 
                      kernel_size    = parms['conv_kernal_size']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = parms['pooling_kernal_size']),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(parms['dnn_input_size'] * ((input_size - 2) // 2), parms['dnn_output_size']),
            nn.ReLU(),
            nn.Linear(parms['dnn_output_size'], num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        x = self.fc_layer(x)
        x = self.softmax(x)
        return x


# Loading the classification layer
model_path = os.path.join(config['classification_layer_path'],
                          config['trained_classification_layer'])

loaded_model = CNN_DNN_Classifier(input_size  = parms['input_size'],
                                  num_classes = parms['num_classes'])

checkpoint = torch.load(model_path)

loaded_model.load_state_dict(checkpoint['state_dict'])

loaded_model.eval()


# Creating a function for sentiment weight genrator
def sentimentAndWeightGenerator(news:str,
                                   timestamp:str)-> pd.DataFrame:
    """This function take news and the timestamp as input and returns its sentiment and weight as a dataframe
    Args:
        news (str): give only one news as input.
        timestamp (str) (YYYY-MM-DD HH:MM:SS): Timestamp of news arrival 
    Returns:
        pd.DataFrame: 
    """
    
    # Slicing the news into smaller parts
    news_ = slice_news(news)
    
    # Preprocessing
    cleaned_news = list_text_cleaning(news_)
    
    ########################## Sentiment Analysis #############################
    # Tokenizing
    tokenized_data = tokenizer.batch_encode_plus(
    cleaned_news,
    add_special_tokens=True,
    max_length= parms['max_words_in_three_sentence'],
    padding= parms['padding'],
    truncation=True,
    return_attention_mask=True,
    return_token_type_ids=False,
    return_tensors='pt')
    
    # BERT output extraction
    inputs = {k:v.to(device) for k,v in tokenized_data.items()} # k : input ids and v: attention mask
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Transfering the tensors in CPU    
    X = outputs.last_hidden_state[:,0].cpu()
    
    # Classifing the sentiment using the loaded classifier model
    with torch.no_grad():
        predictions = loaded_model(X.unsqueeze(1)) # Unsqueeze to fit it into a 1d convolution
        
    # Traforming the tensors into a numpy array    
    predictions_np = predictions.numpy()
    
    # Taking argmax get the labels with maximum probabilities
    predictions_argmax = np.argmax(predictions_np, axis=1)

    # Assuming you have a mapping of class indices to class labels
    class_mapping =  {2: -1, 0: 0, 1: 1} #parms['class_mapping_of_sentiment']  

    # Convert class indices to class labels
    predicted_labels = [class_mapping[idx] for idx in predictions_argmax]  
    
    company_weight_list  = []
    indistry_weight_list = []
    keyword_weight_list  = []
    word_weight_list = []
    bigram_weight_list = []
    for i in range(len(cleaned_news)):
        # Finding company and industry weights
       company_weight, indistry_weight = company_industry_weight_generator(str(cleaned_news[i]),
                                                                           int(predicted_labels[i]))
       company_weight_list.append(company_weight)
       indistry_weight_list.append(indistry_weight)
       
       # Finding keyword weights
       keyword_weight_list.append(keyword_weight_generator(str(news_[i]),
                                                           int(predicted_labels[i])))
       
       word_weight_list.append(unique_word_weight_generator(str(news_[i]),
                                                            int(predicted_labels[i])))
       
       bigram_weight_list.append(unique_bigram_weight_generator(str(news_[i]),
                                                                int(predicted_labels[i])))
       
       
    
    list_of_news_weights = [np.nanmean(company_weight_list),
                            np.nanmean(indistry_weight_list),
                            np.nanmean(keyword_weight_list),
                            np.nanmean(word_weight_list),
                            np.nanmean(bigram_weight_list)]
    
    list_of_weights = parms['news_class_weight_matrix']
       
    # Generating weight for each news input
    final_weights = []
    for i in range(len(list_of_weights)):
        weight_ = list_of_news_weights[i] * list_of_weights[i]
        final_weights.append(weight_)
        
    if np.nanmean(final_weights) != 0:
        sentiment_score = np.nanmean(final_weights)
    else:
        if mode(predicted_labels) < 0:
            sentiment_score = mode(predicted_labels) * parms['minimum_weight_to_assign']
        else:
            sentiment_score = mode(predicted_labels) * parms['minimum_weight_to_assign']

    
    # Getting the news timestamp without the second
    date_ = timestamp[:10]
    time_ = timestamp[11:17]+"00"
    
    # Appending into a dataframe
    output_df = pd.DataFrame({"text": [news],
                              "DATE":[date_],
                              "TIME":[time_],
                              "Sentiment_Score": [sentiment_score]})
    

    return output_df




def sentiment_score_df_maker(df:pd.DataFrame, 
                             date_column  = "Timestamp",
                             news_column = "summary")->pd.DataFrame:
    """
    This function takes a news dataframe as input and calculate sentiment_score to only those news which came during the market hour
    """
    appended_data = []
    df = df.dropna()
    df.reset_index(inplace= True,drop = True)
    for i in range(len(df)):
        if marketTimeFilter(str(df.iloc[i][date_column])) == 'yes':
            # sys.stdout.write(f"\r Processing...... {(i/len(df) *100)} %")
            # sys.stdout.flush()
            news = df.iloc[i][news_column]
            timestamp = df.iloc[i][date_column]
            odf = sentimentAndWeightGenerator(str(news),str(timestamp))
            appended_data.append(odf)
        else:
            pass
    output_df = pd.concat(appended_data)
    output_df = output_df[output_df['Sentiment_Score'] != 0].reset_index(drop = True)
    return output_df.iloc[::-1]
