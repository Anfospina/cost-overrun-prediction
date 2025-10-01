def format_numeric(df, column):
    """
    Convert a column to data integer.
    """
    df[column] = df[column].astype(int)
    return df