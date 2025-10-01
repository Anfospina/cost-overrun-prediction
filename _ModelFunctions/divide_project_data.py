def divide_project_data(data, target=None, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test=train_test_split(
    data.drop(columns=[target]),
    data[target],
    test_size=test_size,
    random_state=random_state
)
    return x_train, x_test, y_train, y_test