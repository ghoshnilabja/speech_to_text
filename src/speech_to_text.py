# Importing Libraries
from datetime import datetime
from psycopg2 import sql
from vosk import Model, KaldiRecognizer
import pyaudio
import re
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import yaml
import os
import re
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from functools import partial
from transformers import AutoTokenizer, BertModel, BertTokenizer
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from transformers import logging
from config import InsertNewsSignal, isHoliday,nifty_companies,FinBERT_Tone_Model_Path,FinBERT_Tone_Model,input_channels,output_channels,conv_kernal_size,pooling_kernal_size
from config import dnn_input_size,dnn_output_size,classification_layer_path,trained_classification_layer,input_size,num_classes,max_words_in_three_sentence,padding,vosk_model_path



logging.set_verbosity_error()

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained BERT model and tokenizer (This is taken from my previous model )
# model_name = FinBERT_Tone_Model
tokenizer = BertTokenizer.from_pretrained(FinBERT_Tone_Model_Path)
bert_model = BertModel.from_pretrained(FinBERT_Tone_Model_Path).to(device)

# Defining classifier model class
class CNN_DNN_Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_DNN_Classifier, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels    = input_channels, 
                    out_channels   = output_channels, 
                    kernel_size    = conv_kernal_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = pooling_kernal_size),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(dnn_input_size * ((input_size - 2) // 2), dnn_output_size),
            nn.ReLU(),
            nn.Linear(dnn_output_size, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        x = self.fc_layer(x)
        x = self.softmax(x)
        return x


# Loading the classification layer
model_path = os.path.join(classification_layer_path,
                        trained_classification_layer)

loaded_model = CNN_DNN_Classifier(input_size  = input_size,
                                num_classes = num_classes)

checkpoint = torch.load(model_path)

loaded_model.load_state_dict(checkpoint['state_dict'])
loaded_model.eval()

def text_sent(text:str):
    # Tokenizing
    tokenized_data = tokenizer.batch_encode_plus(
    [text],
    add_special_tokens=True,
    max_length= max_words_in_three_sentence,
    padding= padding,
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
    
    # print(predicted_labels[0])
    return predicted_labels[0]

# Utils Fun
def find_matching_symbols(args):
    text, keyword, symbol = args
    if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
        return symbol
    return None

def find_matching_symbols_parallel(text, df):
    with ProcessPoolExecutor() as executor:
        args_list = [(text, kw, sym) for kw, sym in zip(df['Company Name'], df['Symbol'])]
        matching_symbols_list = list(executor.map(find_matching_symbols, args_list))

    matching_symbols = set([symbol for symbol in matching_symbols_list if symbol is not None])
    return list(matching_symbols)

# Appending Model
model = Model(vosk_model_path)

# Initiating the sound recognizer
recognizer = KaldiRecognizer(model, 16000)

# # Setting up audio input
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, 
                channels=1, 
                rate=16000, 
                input=True,
                frames_per_buffer=8192)

stream.start_stream()

# Reading the database
symbl_df = pd.read_csv(nifty_companies)

while True:
    if (str(datetime.now().time()) > '15:35:00'):
        break  
    data = stream.read(4096)
    if recognizer.AcceptWaveform(data):
        text = recognizer.Result()
        cleaned_text = text[14:-3]
        sent_ = text_sent(cleaned_text)
        companies_list = find_matching_symbols_parallel(cleaned_text, symbl_df)
        # print("===>",cleaned_text,sent_,companies_list)
        InsertNewsSignal(cleaned_text,sent_,companies_list)


current_date = str(datetime.now().date())
print("current_date",current_date)

# if isHoliday(current_date) == False:
#     speechToText(current_date)

