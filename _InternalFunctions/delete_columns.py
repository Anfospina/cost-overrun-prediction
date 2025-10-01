def delete_columns(df, columns):
    """
    Delete columns from a DataFrame.
    """
    return df.drop(columns=columns)