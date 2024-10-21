import pandas as pd
import holidays
from datetime import datetime
import pickle

def get_holiday_info(date):
    # Convert the string to a datetime object
    # date = datetime.strptime(date_str, '%Y-%m-%d')
    # Get holidays in Norway (you can specify region if needed)
    norwegian_holidays = holidays.Norway(years=date.year)
    if date in norwegian_holidays:
        return norwegian_holidays[date]
    else:
        return None

val = 'data/time_series_val.pkl'
train = 'data/time_series_train.pkl'
test = 'data/time_series_test.pkl'
all_files = [train, val, test]

save_file = {}

for file in all_files:
    print(file)
    df = pd.read_pickle(file)
    dates = list(df.index)
    for date in dates:
        holiday = get_holiday_info(date)
        if holiday:
            save_file[date.strftime('%Y-%m-%d')] = holiday


# Specify the filename
filename = 'data/norway_holidays.pkl'

# Open the file in write-binary mode and save the dictionary
with open(filename, 'wb') as file:
    pickle.dump(save_file, file)