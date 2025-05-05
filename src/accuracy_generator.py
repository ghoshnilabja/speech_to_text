import pandas as pd
from datetime import datetime, timedelta, time, date
import numpy as np
# import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, time
import numpy as np

# Creating a function that creats previous/future time stamp
def timeGenerator(time:str,lookback:int)->str:

    """This function generates time accroding to the lookback.
       For example, if input is 12:28:00 and lookback = 12 
                    then output is 12:16:00

    Args:
        time (str): in HH-MM-SS format
        lookback (int): durition in minute to get the past data 

    Returns:
        str: past timestamp in HH-MM-SS format
    """
    try:
        input_time = datetime.strptime(time, '%H:%M:%S')
        time_difference = timedelta(minutes=lookback)
        new_time = input_time - time_difference
        past_time = str(new_time.strftime('%H:%M:%S'))[:6]+"00"
        return past_time
    except ValueError:
        return "Invalid time format. Please use HH:MM:SS."


def accuracyGenerator(signal_df:pd.DataFrame,
                       signal_time: str,
                       signal: str,
                       nifty_df: pd.DataFrame,
                       nifty_df_time: str,
                       open: str,
                       time_to_check_accuracy:int) -> pd.DataFrame:
    """This function take signal_df and nifty50 spot dataframe as input and returns the signal accuracy accroding to the time to check.
   
    Args:
        signal_df (pd.DataFrame): output of sentiment_signal_generation function
        signal_time (str): Time as HH-MM-SS format
        signal (str): Signal Column Name
        nifty_df (pd.DataFrame): Nifty 50 Spot Data
        nifty_df_time (str): Time column name
        open (str): OPEN column name
        time_to_check_accuracy (int): Time want to check the accuracy

    Returns:
        pd.DataFrame: Dataframe contains date and accuracy
    """    
    try:

        # Dropping duplicates values (if any)
        df = signal_df.drop_duplicates()
        # Resetting the index
        df = df.reset_index(drop= True)
        true_signal  = 0
        false_signal = 0
        total_pl     = 0
        for i in range(len(df)):
            time_ = str(df[signal_time].iloc[i])[:5]+':00'
            time_to_check = (str(timeGenerator(time_,time_to_check_accuracy * -1)))
            if datetime.strptime(time_to_check, '%H:%M:%S').time() < time(15,30,00):
                time_to_check = time_to_check
            else:
                time_to_check = "15:30:00"
            try:
                current_open = nifty_df[nifty_df[nifty_df_time] == time_][open].values[0]
                future_open  = nifty_df[nifty_df[nifty_df_time] == time_to_check][open].values[0]
                difference = future_open - current_open
                signal_ = int(df[signal].iloc[i])  
                val_ = difference * signal_
                total_pl += val_
                if val_ >= 0:
                    true_signal += 1
                else:
                    false_signal += 1
            except:
                pass
        if true_signal + false_signal == 0:
            accuracy = None
        else:
            accuracy = true_signal/(true_signal + false_signal)
        return accuracy, total_pl
    except Exception as error:
        pass




def minuteAccuracyGenerator(signal_dataframe:pd.DataFrame,
                             nifty_data:pd.DataFrame)->dict:
    """This fucntion will generates three types of accuracies for the signal dataframe
    Args:
        signal_df (pd.DataFrame): A particular days signal dataframe
    Returns:
        dict: A dictionary with different types of accuracies
    """
    # Previous Results
    initial_3_min_acc = None
    initial_6_min_acc = None
    initial_9_min_acc = None
    initial_12_min_acc= None
    initial_15_min_acc = None
   
    dict_of_min_pl = {'3':None, '6':None,'9':None, '12':None, '15':None}
   
    if len(signal_dataframe) > 0:
        try:
            signal_df = signal_dataframe.reset_index()
            nifty = nifty_data
            signal_df_timestamp_col = 'timestamp'
            # Creating date and time column for signal df

            signal_df['DATE'] = signal_df[signal_df_timestamp_col].apply(lambda x : str(x)[:10])
            signal_df['TIME'] = signal_df[signal_df_timestamp_col].apply(lambda x : str(x)[11:19])
            minutes_to_check = [3,6,9,12,15]
            list_of_min_accs = []
            for min in minutes_to_check:
                acc, pl = accuracyGenerator(signal_df,
                            signal_time = 'TIME',
                            signal = 'result',
                            nifty_df = nifty,
                            nifty_df_time = 'TIME',
                            open = 'OPEN',
                            time_to_check_accuracy = min)
               
                list_of_min_accs.append(acc)
                dict_of_min_pl[str(min)] = round(pl,2)
                if min == 3:
                    initial_3_min_acc = round(acc * 100, 2)
                elif min == 6:
                    initial_6_min_acc = round(acc * 100 , 2)
                elif min == 9:
                    initial_9_min_acc = round(acc* 100 , 2)
                elif min == 12:
                    initial_12_min_acc = round(acc* 100, 2)
                elif min == 15:
                    initial_15_min_acc = round(acc * 100, 2)
            output = {"Total_Signals":len(signal_df),
                    "3": round((list_of_min_accs[0] * 100),2),
                    "6": round((list_of_min_accs[1] * 100),2),
                    "9": round((list_of_min_accs[2] * 100),2),
                    "12": round((list_of_min_accs[3] * 100),2),
                    "15": round((list_of_min_accs[4] * 100),2)
                    }
            return output, dict_of_min_pl
        except Exception as error:
            output = {"Total_Signals":len(signal_df),
                    "3": initial_3_min_acc,
                    "6": initial_6_min_acc,
                    "9": initial_9_min_acc,
                    "12": initial_12_min_acc,
                    "15": initial_15_min_acc
                    }
            return output, dict_of_min_pl
    else:
        output = {"Total_Signals":len(signal_dataframe),
                    "3": initial_3_min_acc,
                    "6": initial_6_min_acc,
                    "9": initial_9_min_acc,
                    "12": initial_12_min_acc,
                    "15": initial_15_min_acc
                    }
        return output, dict_of_min_pl
