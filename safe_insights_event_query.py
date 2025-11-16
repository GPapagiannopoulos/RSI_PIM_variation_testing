'''
File name: safe_insights_event_query.py 
Author: George Papagiannopoulos 
Email Address: george.papagiannopoulos@rssb.co.uk
Description: This function queries the Safe Insights database to extract event data based on the PIM query found in the SQL file provided. 
Last Edited: George Papagiannopoulos 13/11/2025
'''
import sqlalchemy
import pandas as pd 
import os

def extract_safe_insights_data(safe_insights_connection_details:dict, query_path:str) -> pd.DataFrame:
    '''
    This is a simple function that initializes a connection to the Safe Insights database using SQLAlchemy 
    and extracts the data based on the PIM query. The data is then returned as a pandas DataFrame and exported to a CSV file.
    '''
    # Create the connection URL using own credentials
    connect_url = sqlalchemy.engine.URL.create(
        "postgresql+psycopg2",
        username = safe_insights_connection_details["username"],
        password = safe_insights_connection_details["password"],
        host = safe_insights_connection_details["host"], 
        port = safe_insights_connection_details["port"], 
        database = safe_insights_connection_details["database"]
    )

    # Create engine using the connection URL created in the previous step
    si_engine = sqlalchemy.create_engine(connect_url) 

    # Retrieve SQL query 
    query_file_path = query_path

    # Use query and engine to extract data from Safe Insights
    with si_engine.connect() as con:
        with open(query_file_path) as file:
            query = sqlalchemy.text(file.read())
            si_data = pd.read_sql(query, con = si_engine, 
                            dtype = {
                                "YYYYMMDD" : 'int',
                                "event_date" : 'datetime64[s]',
                                "smis_reference": "string", 
                                "description": "string", 
                                "RANK": 'int', 
                                "summary_cause": "string",
                                "detailed_cause": "string"
                                }
                            )
            

    return si_data

def generate_safe_insights_data(query_path:str) -> pd.DataFrame:
    
    safe_insights_connection_details = {
        "username" : os.getenv("si_username"),
        "password" : os.getenv("si_password"),
        "host" : os.getenv("si_host"), 
        "port" : os.getenv("si_port"), 
        "database" : os.getenv("si_database")
    }
    safe_insights_data = extract_safe_insights_data(safe_insights_connection_details, query_path)
   
    safe_insights_data.to_csv("safe_insights_extract.csv")
    print("Safe Insights and SPAD calculations completed and extracted.")
    return safe_insights_data