# Importing the libraries
import pandas as pd
from constants import dataframe_constants as dfc
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


def calc_progress_percent(dataframe, value, iterate_by=1):
    num_rows = len(dataframe.index)
    group = int((num_rows * iterate_by) / value)

    return group


# Delete columns that we dont need
def delete_columns(dataframe):
    dataframe = dataframe.drop(columns=dfc.useless_column)
    dataframe = final_parse(dataframe)
    return dataframe


# Label the data with 1 and 0 (1 for malignant, 0 for benign)
def label_data(dataframe, progress_bar):
    dataframe.insert(loc=len(dataframe.columns), column='severity', value='0')

    value = progress_bar.value()
    group = calc_progress_percent(dataframe, value=value)
    count = 0

    for (index, row) in dataframe.iterrows():
        count += 1
        if count == group:
            value += 1
            progress_bar.setValue(value)
            count = 0
        for keyword in dfc.keywords_agent:
            if keyword in row['useragent'].lower():
                dataframe.at[index, 'severity'] = '1'

                for keyword in dfc.keywords_response:
                    if keyword in str(row['response_code']):
                        dataframe.at[index, 'severity'] = '2'

                        for keyword in dfc.keywords_url:
                            if keyword in row['url_path']:
                                dataframe.at[index, 'severity'] = '3'

        if "GET" not in row['http_request_type']:
            dataframe.at[index, 'severity'] = str(int(dataframe.at[index, 'severity']) + 1)

    return dataframe

# def catgroove(dataframe):
#     # Define train and target
#     target = dataframe[['severity']].astype(int)
#     train = dataframe.drop('severity', axis=1)

#     # Define catboost encoder
#     cbe_encoder = ce.cat_boost.CatBoostEncoder()

#     # Fit encoder and transform the features
#     train.astype(str)
#     train_cbe = cbe_encoder.fit_transform(train, target)

#     # Combine back both x and y into one dataframe
#     dataframe = pd.concat([train_cbe, dataframe['severity']], axis=1)

#     return dataframe

def label_encoder(dataframe:pd.DataFrame):
    le = LabelEncoder()
    for column in dataframe.columns:
        dataframe[column] = le.fit_transform(dataframe[column])

    return dataframe

def sample(dataframe):
    dataframe = dataframe.groupby('severity').sample(n=10000, random_state=0, replace=True)
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

def final_parse(dataframe):
    # new data frame with split value columns
    new = dataframe["url"].str.split(" ", n=1, expand=True)

    # making separate first name column from new data frame
    dataframe["http_request_type"] = new[0]

    # making separate last name column from new data frame
    dataframe["url_path"] = new[1]

    # Dropping old Name columns
    dataframe.drop(columns=["url"], inplace=True)
    return dataframe
