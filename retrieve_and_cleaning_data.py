"""
Created on Monday Sep. 12-2022.

@author: Vasco Schiavo
This script aims to take the raw data made available by the Swiss Federal Railways about the punctuality of trains and
clean it. This data is used to train a train delay prediction model. The data about trains' punctuality is available
daily on the website: data.sbb.ch. The script accesses this webpage, downloads the data in csv format, and then
processes it.
"""

# Packages and modules
import requests
import calendar
from datetime import date
import pandas as pd


def time_in_seconds(time):
    """
    The method implements the conversion of a time in seconds
    :param time: string of the form 'HH:MM:SS', where H, M and S are digits
    :return time_sec: integer representing the time in seconds
    """

    time_sec = int(time[0:2])*3600 + int(time[3:5])*60 + int(time[6:8])

    return time_sec

def date_to_nb(day_date):
    """
    The method assigns to every day of the week a number. If the date corresponds to a holiday, it assigns a special
    number.
    :param day_date: string of the form AAAA-MM-DD, where A, M and D are integers.
    :return: integer between 1 and 8.
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
    This method accesses to the web-site: https://data.sbb.ch
    and download the .csv file which contains the train delay of yesterday. This .csv file is saved with
    name: yesterday_date.csv.
    :param url: string representing the url to download the data
    :param name_download: string representing the name to give to the downloaded file
    """

    # Download
    response = requests.get(url)
    open(name_download + ".csv", "wb").write(response.content)

def selecting_and_cleaning_data(imported_data):
    """
    This method takes as input the raw data imported from the csv file and select the interesting columns for the
    project. After that, it cleans tha data erasing NaN values and rewriting some entries.
    :param imported_data: DataFrame
    :return data: DataFrame
    """

    # Selects columns to be used in the model's training
    data = imported_data.loc[:,['Day of operation', 'Linie', 'Stop name', 'Departure time', 'Departure forecast']]

    # Drops rows with NaN values. If a row contains a NaN values, it means that the train is canceled. Therefore,
    # there is no delay ;)
    data = data.dropna()

    # Renames index
    data.index = pd.RangeIndex(len(data.index))

    delay_list = list()
    stop_names_OPUIC = pd.read_csv('stop_names_OPUIC.csv', ',', index_col = 0)

    # Cleans the data
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

    # Returns data without 'Departure forecast' column
    return data.loc[:,['Day of operation', 'Linie', 'Stop name', 'Departure time', 'Delay']]

########################
#   MAIN
########################

# Parameters to download the data
yesterday = date.today() - timedelta(days=1)
yesterday_date = yesterday.strftime('%d.%m.%Y')
url = "https://data.sbb.ch/explore/dataset/actual-data-sbb-previous-day/download/?format=csv&timezone=Europe/Berlin&lang=en&use_labels_for_header=true&csv_separator=%3B"

#Download the data from data.SBB.ch website and saves it on our machine
download_and_save(url, yesterday_date)

# Imports the whole data from csv file
imported_data = pd.read_csv('ist-daten-sbb.csv', ';')

# Cleans imported_data
data = selecting_and_cleaning_data(imported_data)

# Adds new data to the existing one
train_delay_data = pd.read_csv('/home/camilla/PycharmProjects/pythonProject/train_delay_data.csv', index_col=0)
train_delay_data = pd.concat([train_delay_data,data], ignore_index=True)

# Saves data in a .csv file
data.to_csv( '/home/camilla/PycharmProjects/pythonProject/train_delay_data.csv')
