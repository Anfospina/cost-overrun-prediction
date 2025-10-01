import pandas as pd
def format_date(df, column):
    """
    Convert a column to datetime.
    """
    df[column] = pd.to_datetime(df[column])
    return df