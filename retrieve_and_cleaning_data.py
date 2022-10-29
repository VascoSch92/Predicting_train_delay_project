"""
Created on Monday Sep. 12-2022.

@author: Vasco Schiavo

This script downloads the data made available by the Swiss Federal Railways about the punctuality of trains and cleans
it. The cleaned data is used to train a train delay prediction model. The train punctuality data is available daily on
the website: http//data.sbb.ch.
"""

# Packages and modules
import requests
import calendar
from datetime import date, timedelta
import pandas as pd


def time_in_seconds(time):
    """
    The method implements the conversion of a time in seconds.

    Parameters
    ----------
    time: string
        The string is of the form 'HH:MM:SS', where H, M and S are digits.
        
    Return
    ------
    time_sec: int
        Integer representing the time in seconds.
    """

    time_sec = int(time[0:2])*3600 + int(time[3:5])*60 + int(time[6:8])

    return time_sec

def date_to_nb(day_date):
    """
    The method assigns to every day of the week a number. If the date corresponds to a holiday, it assigns a special
    number.

    Parameters
    ----------
    day_date: string
        The string is of the form AAAA-MM-DD, where A, M and D are integers.
        
    Return
    ------
    dict_day_nb["Holiday"]: int
        Integer between 1 and 8
    """

    # Dictionary describing the map between days and integers
    dict_day_nb = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7,
                   "Holiday": 8}

    # List of swiss holidays
    swiss_holiday_list = ["01-01",  # New Years Day
                          "01-15",  # Good Friday
                          "05-26",  # Ascension Day
                          "08-01",  # National Day
                          "12-25",] # Christmas

    # Cases where the day_date corresponds to a swiss holiday
    if day_date[5:] in swiss_holiday_list:
        return dict_day_nb["Holiday"]

    # Otherwise
    else:
        # Day corresponding to this date
        day = calendar.day_name[date(int(day_date[0:4]), int(day_date[5:7]), int(day_date[8:10])).weekday()]
        return dict_day_nb[day]


def download_and_save(url, name_download):
    """
    This method accesses to the web-site: https://data.sbb.ch and download the .csv file which contains the train
    delay of yesterday. This .csv file is saved with name: train_delay_data_yesterday_date.csv.

    Parameters
    ----------
    url: string
        String representin the url to download the cleaned_data.
    name_download: string
        String representing the name to give to the downloaded file.
    """

    # Download
    response = requests.get(url)
    open(name_download + ".csv", "wb").write(response.content)

def translation(data):
    """
    This method translate the columns which we will use in the model from german to english.

    Parameters
    ----------
    data: dataframe
        Dataframe downloaded in german language.

    Return
    ------
    cleaned_data: dataframe
        Dataframe translated to english.
    """

    data.rename(columns={'Betriebstag': 'Day of operation',
                         'Haltestellen Name': 'Stop name',
                         'Abfahrtszeit': 'Departure time',
                         'Ab Prognose': 'Departure forecast',
                         }, inplace=True)

    return data

def selecting_and_cleaning_data(imported_data):
    """
    This method takes as input the raw data imported from the csv file and select the interesting columns for the
    project. After that, it cleans them erasing NaN values and rewriting some entries.

    Parameters
    ----------
    imported_data: Dataframe
    
    Return
    ------
    cleaned_data: Dataframe
    """

    # Drops rows with NaN values. If a row contains a NaN values, it means that the train is canceled. Therefore,
    # there is no delay ;)
    data = imported_data.dropna()

    # Renames index
    data.index = pd.RangeIndex(len(data.index))

    delay_list = list()
    stop_names_OPUIC = pd.read_csv('stop_names_OPUIC.csv', ',', index_col = 0)

    # Cleans the cleaned_data
    for idx, row in data.iterrows():

        # Entry 'Day of operation' is converted in a number
        data.at[idx, 'Day of operation'] = date_to_nb(data.at[idx, 'Day of operation'])

        # To every 'Stop name' is assigned a number
        temp_stop_name = stop_names_OPUIC[
            stop_names_OPUIC['Stop name'] == data.at[idx, 'Stop name']].index
        if len(temp_stop_name) == 0:
            data.at[idx, 'Stop name'] = -1
        else:
            data.at[idx, 'Stop name'] = temp_stop_name[0]

        # 'Departure time' and 'Departure forecast' entries are cleaned and converted in seconds
        data.at[idx, 'Departure time'] = time_in_seconds(data.at[idx, 'Departure time'][11:])
        data.at[idx, 'Departure forecast'] = time_in_seconds(data.at[idx, 'Departure forecast'][11:])

        # Computation of the delay (if it is the case)
        delay_list.append(max(data.at[idx, 'Departure forecast'] - data.at[idx, 'Departure time'], 0))

    # Adds a column containing the delay in second
    data['Delay'] = delay_list

    # Returns cleaned_data without 'Departure forecast' column
    return data.loc[:,['Day of operation', 'Linie', 'Stop name', 'Departure time', 'Delay']]

########################
#   MAIN
########################

# Parameters to download the data
yesterday = date.today() - timedelta(days=1)
yesterday_date = yesterday.strftime('%d.%m.%Y')
url = "https://data.sbb.ch/explore/dataset/ist-daten-sbb/download/?format=csv&timezone=Europe/Berlin&lang=de&use_labels_for_header=true&csv_separator=%3B"

# Downloads the raw data from data.SBB.ch and saves it in the raw_data folder
download_and_save(url, f"/Users/argo/PycharmProjects/Train_Delay_Predicion/raw_data/{yesterday_date}")

# Reads the raw data
imported_data = pd.read_csv(f"/Users/argo/PycharmProjects/Train_Delay_Predicion/raw_data/{yesterday_date}.csv",
                            delimiter=';',
                            usecols=['Betriebstag', 'Linie', 'Haltestellen Name', 'Abfahrtszeit', 'Ab Prognose'])

# Translation from German to English
imported_data = translation(imported_data)

# Cleans imported_data
data = selecting_and_cleaning_data(imported_data)

# Saves cleaned_data as a CSV file in the cleaned_data folder
data.to_csv( f"/Users/argo/PycharmProjects/Train_Delay_Predicion/cleaned_data/train_delay_data_{yesterday_date}.csv")
