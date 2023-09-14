import pandas as pd
from pandas import DataFrame

import CONST


def save_csv(df: DataFrame, file_path: str):
    return df.to_csv(file_path, index=False)


def filter_empty_columns(df: DataFrame, colun_name):
    return df[df[colun_name].notnull() & (df[colun_name] != '')]


def num_overlapped_values(df: DataFrame, column_name: str):
    return (df[column_name].value_counts() > 1).sum()


if __name__ == '__main__':
    # Read csv
    df = pd.read_csv(CONST.ANNOTATION_PATH)

    # Filter into selected columns
    df = df.filter(items=CONST.SELECT_COLUMN)

    # Filter some row with empty values
    df = filter_empty_columns(df, CONST.WOUND_TYPE)
    df = filter_empty_columns(df, CONST.WOUND_LOCATION)

    # Drop the duplication
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[CONST.FILE_NAME], keep="first")

    # Check the number of overlapped file
    print(f"Number of overlapped file: {num_overlapped_values(df, CONST.FILE_NAME)}")

    # Save csv
    save_csv(df, CONST.ANNOTATION_PROCESSED_PATH)
