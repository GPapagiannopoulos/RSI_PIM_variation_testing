'''
File name: safe_insights_event_counts.py 
Author: George Papagiannopoulos 
Email Address: george.papagiannopoulos@rssb.co.uk
Description: This function compares the number of events by precursor reported into Safe Insights since 2017. 
Last Edited: George Papagiannopoulos 13/11/2025
'''

import pandas as pd 
import safe_insights_event_query as sieq
import os 

if not os.path.exists("safe_insights_extract.csv"):
    print("Data not found - Executing query...")
    sieq.generate_safe_insights_data(query_path = "All_events_query.sql")
else:
    print("Data available - Loading data now")
safe_insights_data = pd.read_csv("safe_insights_extract.csv")

def group_by_period(dataframe:pd.DataFrame) -> pd.DataFrame:
    '''
    We want to compare results by period. To do this we need the period end dates 
    which are saved as a csv file. 
    '''
    # Creating a skeleton of all precursor id x period combinations 
    period_end_dates = pd.read_csv("period_end_dates.csv")
    period_end_dates["End Date"] = pd.to_datetime(period_end_dates["mmm yyyy"], format="%Y-%m-%d")
    period_end_dates.drop(columns = ["mmm yyyy", "mmm yyyy"], inplace=True)
    # Converting the period column from a categorical to two numerical columns for easier sorting and manipulation
    period_end_dates["year"] = period_end_dates["End Date"].dt.year
    period_end_dates["annual_period"] = period_end_dates["Period"].astype(str).str.split("P").str[1].astype(int)
    
    dataframe["event_date"] = pd.to_datetime(dataframe["event_date"], format="%Y-%m-%d")
    dataframe = dataframe[dataframe["event_date"] <= pd.to_datetime("2025-11-08")]
    dataframe = pd.merge_asof(dataframe, period_end_dates, left_on="event_date", right_on="End Date", direction="forward")

    grouping = dataframe.groupby(["End Date", "year", "annual_period", "Period", "PREC_ID"], as_index=False)["smis_reference"].count()

    grouping.to_csv("group.csv")

    return grouping 

    
