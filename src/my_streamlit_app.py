# import streamlit as st
# import plotly.graph_objects as go
import pandas as pd

config = {'displaylogo': False,
          'modeBarButtonsToAdd': ['drawline',
                                  'drawopenpath',
                                  'drawrect',
                                  'eraseshape'],
          'modeBarButtonsToRemove': ['lasso2d', 'select2d']}


# Toggle filter function
def toggle_filter_fun(signal_dataframe: pd.DataFrame,
                      signal_col='result',
                      time_stamp_col='timestamp') -> pd.DataFrame:
    """This function filters the dataframe if the same signal came concurrently
    Args:
        signal_dataframe (pd.DataFrame): 
        signal_col (str, optional): Defaults to 'result'.
    Returns:
        pd.DataFrame: Filtered DataFrame
    """

    # Sorting by the timestamp column
    signal_dataframe = signal_dataframe.sort_values(by=time_stamp_col)

    # Identify rows to be removed using boolean indexing
    mask = ((signal_dataframe[time_stamp_col].shift(1).dt.date == signal_dataframe[time_stamp_col].dt.date) &
            (signal_dataframe[signal_col].shift(1) == signal_dataframe[signal_col]))

    # Invert the mask to keep the rows you want to keep
    signal_dataframe = signal_dataframe[~mask]

    # Reset the index of the resulting DataFrame
    signal_dataframe = signal_dataframe.reset_index(drop=True)
    return signal_dataframe



def toggleFilterFun(dictionary):
    new_dictionary = {}
    previous_value = None

    for key, value in dictionary.items():
        if value != previous_value:
            new_dictionary[key] = value
        previous_value = value

    return new_dictionary


