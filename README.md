# speech_to_text
# Nifty_Sentiment_Live VERSION: 2

# Create conda env:
conda create --n live_sentiment python==3.11

#------------------- LIVE_NEWS_SIGNALS ----------------------------#
SHELL=/bin/bash
BASH_ENV=~/.bashrc_conda
44 09 * * 1-5 source /home/vista-ai/anaconda3/etc/profile.d/conda.sh && conda activate live_sentiment && /home/vista-ai/anaconda3/envs/live_sentiment/bin/python /home/vista-ai/MODELS/sentiment-live_feedsense/src/automator.py >> /home/vista-ai/MODELS/sentiment-live_feedsense/crontab_logs/reports.log 2>&1

SHELL=/bin/bash
BASH_ENV=~/.bashrc_conda
16 09 * * 1-5 source /home/vista-ai/anaconda3/etc/profile.d/conda.sh && conda activate live_sentiment && /home/vista-ai/anaconda3/envs/live_sentiment/bin/python /home/vista-ai/MODELS/sentiment-live_feedsense/src/run.py >> /home/vista-ai/MODELS/sentiment-live_feedsense/crontab_logs/run.log 2>&1




#-----------------------------------------------------------------------------------------------------------------------------------------#


# Nifty_Sentiment_Live VERSION: 1
Sentiment Signal Model for NIFTY 50

### Please Note:
The to run the speech_to_text.py the RNN based Vosk model needs around 18 GB of RAM


conda create --name sentiment_env python=3.11.5
conda activate sentiment_env

pip3 install psycopg2-binary
conda install conda-forge::pyaudio==0.2.14
