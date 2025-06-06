def rename_columns(df, rename_dict):
    """
    Rename columns in a DataFrame.
    """
    return df.rename(columns=rename_dict)  