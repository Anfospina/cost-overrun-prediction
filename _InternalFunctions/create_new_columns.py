def create_new_columns(df, new_col_name,operation):
    """
    Create a new column based on existing columns.
    """
    df[new_col_name] = operation
    return df