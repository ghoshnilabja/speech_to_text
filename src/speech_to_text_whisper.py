import whisper
import pyaudio
import numpy as np
import psycopg2
from psycopg2 import sql
import pandas as pd
import os, re, wave
import yaml, torch, spacy
from torch import nn
from transformers import BertTokenizer, BertModel
from collections import deque
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import warnings; warnings.filterwarnings('ignore')
from sentiment_analysis import analyze_sentiment
from langdetect import detect
from ast import literal_eval 
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp_gib = spacy.load("en_core_web_sm")


# Audio configurations
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK_SIZE = 4096
DURATION_SECONDS = 15

# Load config files
def load_config(config_name, CONFIG_PATH):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        return yaml.safe_load(file)

def lexical_diversity(text):
    words = text.split()
    unique_words = set(words)
    if len(words) == 0:
        return 0
    else:
        return len(unique_words) / len(words)

def is_repetitive(text):
    if lexical_diversity(text= text) < 0.15:
        return True
    else:
        return False

CONFIG_PATH = "/home/nilabja/Documents/nilabja/NLP/nifty_sentiment_live/config"
config = load_config("config.yaml", CONFIG_PATH)
parms = load_config("model_parameters.yaml", CONFIG_PATH)

# Load Whisper model
whisper_model = whisper.load_model("base") #base


def is_gibberish(text):
    try:
        # Check if the text is empty or too short
        if len(text.strip()) < 5:
            return True  # Too short to be meaningful
        
        # Detect if text is English (language detection can sometimes be unreliable for short texts)
        lang = detect(text)
        if lang != "en":
            return True  # If not English, consider it gibberish
        
        # Check for excessive special characters
        special_chars = len(re.findall(r'[^\w\s]', text))
        if special_chars > len(text) * 0.2:  
            return True  # More than 20% of the text is special characters
        
        # Check for excessive word repetition
        words = text.split()
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.4:  # If fewer than 40% of words are unique
            return True
        
        # Process the text with Spacy NLP
        doc = nlp_gib(text)

        # Count meaningful words (words that are not stopwords/punctuation)
        meaningful_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        if len(meaningful_words) < 5:  # If too few meaningful words, likely gibberish
            return True

        # Check if it contains valid named entities (at least one OR has valid sentence structure)
        if len(doc.ents) == 0 and len(words) > 15:  
            return False  # No named entities, but long enough to be meaningful

    except Exception as e:
        print(f"Error: {e}")  # Debugging output
        return True  # If any error occurs, assume gibberish

    return False

# PostgreSQL connection
conn = psycopg2.connect(
    host="localhost", port="5432", user="postgres", password="password", database="nlp"
)
cursor = conn.cursor()

# Load company data
symbl_df = pd.read_csv("/home/nilabja/Documents/nilabja/NLP/nifty_sentiment_live/data/nifty_companies.csv")

# PyAudio setup
mic = pyaudio.PyAudio()
stream = mic.open(
    format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=CHUNK_SIZE
)



def save_audio(frames):
    """Saves collected audio frames to a WAV file."""
    os.makedirs("audio", exist_ok=True)  # Ensure 'audio' directory exists
    filename = "audio/last_chunk_audio.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(mic.get_sample_size(pyaudio.paInt16))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(frames))
    return filename




def transcribe_audio(frames):
    """Convert audio frames to text using Whisper."""
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
    
    result = whisper_model.transcribe(audio_np, fp16=False)
    return result["text"]


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

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


try:
    while True:
        print("Listening...", datetime.now())
        frames = []
        for _ in range(int(AUDIO_RATE / CHUNK_SIZE * DURATION_SECONDS)):
            frames.append(stream.read(CHUNK_SIZE))
        
        print("\nProcessing audio chunk...", datetime.now())
        # audio_file = save_audio(frames)
        transcription = transcribe_audio(frames)

        if not is_repetitive(transcription):
            if not is_gibberish(transcription):

                print("Transcription:", datetime.now(), transcription)
                # try:
                sent_ = nlp(transcription)
                print("Sentiment:", sent_[0]['label'])
                sentiment_mapping = {"Positive": 1, "Negative": -1, "Neutral": 0}
                sent_ = sentiment_mapping.get(sent_[0]['label'], 0)  # Default to 0 if the key is not found

            
                # Match companies
                companies_list = find_matching_symbols_parallel(transcription, symbl_df)
                print("Company list:", datetime.now(), companies_list)

                # Insert into PostgreSQL
                cursor.execute(
                    sql.SQL('''INSERT INTO tbl_news_sentiment_details_finbert (news, sentiment, symbol)
                                VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;'''),
                    (transcription, sent_, companies_list)
                )
                conn.commit()
                    
                # except:
                #     pass

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    mic.terminate()
    conn.close()