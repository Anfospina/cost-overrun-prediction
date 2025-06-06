def delete_records(df, condition):
    """
    Delete records from a DataFrame based on a condition.
    """
    return df[~condition]