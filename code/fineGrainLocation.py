import re
from collections import Counter
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_data(location_data):
    # Delete unwanted columns
    location_data = location_data.drop(['Latitude', 'Longitude'], axis=1)
    # Create new columns for storing labels and date, time
    location_data["Label"] = ""
    location_data['Date'] = pd.to_datetime(location_data['Timestamp']).dt.date
    location_data['Time'] = pd.to_datetime(location_data['Timestamp']).dt.time
    # With multiple locations
    multiple_Loc = location_data.iloc[range(len(location_data))]['Location']
    return multiple_Loc


# split the data by comma
def split_location(multiple_Loc):
    locations_list = []
    locations_list.append(multiple_Loc)
    return locations_list


# Combine all locations into a list of lists
def combine_locations(multiple_Loc):
    list_of_locs = []
    for location in multiple_Loc:
        list_of_locs.append(split_location(location))
    return list_of_locs


# The functions to tokenize the locations
def tokenize(locations):
    words = []
    for l in locations:
        words.extend([i.split(',') for i in l])
    return words


def word_extraction(location):
    keywords = ['University', 'College']
    words = re.sub("[^\w]", " ", location).split()
    cleaned_text = [w for w in words if w in keywords]
    if cleaned_text == []:
        cleaned_text = ['Other']
    return cleaned_text


# This is the function where labels are categorized, this can be changed if needed
def label_generator_words(keyword, label):
    # determine which labels to use
    if 'University' in keyword or 'School' in keyword or 'College' in keyword:
        label.append('Campus')
    elif 'Arana' in keyword:
        label.append('Home')
    elif 'Street' in keyword or 'Road' in keyword or 'Motorway' in keyword:
        label.append('Road')
    elif 'Countdown' in keyword or 'Liquorland' in keyword or 'Supermarket' in keyword or 'Mall' in keyword or \
            'Shop' in keyword:
        label.append('Market')
    elif 'Stables' in keyword or 'Theatre' in keyword or 'Museum' in keyword or 'Club' in keyword or \
            'Societies' in keyword or 'Cafe' in keyword:
        label.append('Leisure')
    else:
        label.append('Others')


# Some of the locations need to be specified manually, according to their frequency of occurrences
def label_generator_numbers(keyword, label):
    if 'Hawthorn Avenue' in keyword or 'Chapel Street' in keyword or 'Clyde Street' in keyword or 'Bay View Road' in keyword or \
            'Shand Street' in keyword or 'Leith Street' in keyword or 'Harrow Street' in keyword:
        label.append('Home')
    elif 'Albany Street' in keyword:
        label.append('Leisure')
    elif 'Union Street' in keyword:
        label.append('Campus')
    else:
        label.append('Others')


def make_histogram(list, xlabel, ylabel, title, direction):
    indices = np.arange(len(list))
    word, frequency = zip(*list)
    plt.bar(indices, frequency)
    plt.xticks(indices, word, rotation=direction)
    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def retrieve_10_locations_with_numbers(list, labels, location_data):
    count_numlist = Counter(list)
    most_count_numlist = count_numlist.most_common(10)
    make_histogram(most_count_numlist, "Location as Numbers", "Occurrences", "Locations that start with numbers",
                   'vertical')

    # Here I made a horizontal bar plot
    location = []
    occurrence = []
    for a, b in most_count_numlist:
        location.append(a)
        occurrence.append(b)
    plt.barh(location, occurrence)
    plt.title("10 Most occurred Locations starting with a number")
    plt.xlabel("Occurrences")
    plt.show()

    labels = [x[0] for x in labels]
    location_data['Label'] = labels

    # Retrieve most common 6 from all Locations
    count_occurrences_list = Counter(labels)
    mostOcc = count_occurrences_list.most_common(6)
    df = pd.DataFrame.from_dict(count_occurrences_list, orient='index').reset_index()
    df = df.rename(columns={'index': 'event', 0: 'count'})

    df.to_csv('count_Occ.csv', index=False)
    # Make a histogram from the six most occurred locations
    make_histogram(mostOcc, "Locations", "Occurrences", "The 6 most occurred occurrences", 'horizontal')


def set_labels(multiple_Loc, location_data):
    list = []
    combi = combine_locations(multiple_Loc)
    tokenized_list = tokenize(combi)
    labels = []
    # For each row inside the 'Location" column
    for l in tokenized_list:
        # Consider everything what is before the first comma
        keyword = l.pop(0)
        words = re.sub("[^\w]", " ", keyword).split()
        # Consider digits first
        if words[0][0].isdigit():
            number = l.pop(0)
            list.append(number + " " + keyword)
            label_generator_numbers(number, labels)
        else:
            label_generator_words(words, labels)

    # Retrieve most common 10 from Locations starting with numbers
    retrieve_10_locations_with_numbers(list,labels, location_data)


# Here I define the frequencies
def convert_to_date(location_data):
    string_date_time = [str(x) for x in location_data['Timestamp']]
    tmp1 = [None] * len(string_date_time)
    i = 0
    for date_time in string_date_time:
        separated_date_time = re.sub("[^\w]", " ", date_time).split()
        minute = int(separated_date_time[4])
        if minute < 30:
            separated_date_time[4] = '00'
            separated_date_time[5] = '00'
        else:
            separated_date_time[4] = '30'
            separated_date_time[5] = '00'
        s = '-'
        t = ':'
        date = separated_date_time[0:3]
        time = separated_date_time[3:6]
        date = s.join(date)
        time = t.join(time)
        new_date_time = date + " " + time
        tmp1[i] = new_date_time
        i = i + 1
    location_data['Intervals'] = tmp1
    location_data.to_csv('convertToDate.csv', index=False)


def write_to_file(data):
    file = open("locations.txt", "w+")
    for i in range(len(data)):
        file.write(str(data['date'][i]) + " " + str(data['Highest_Occ'][i]) + " " + str(data['color'][i]) + "\n")
    file.close()


# Make the color plot by assigning a number to each letter
def determine_color(label):
    if label == 'N':
        return 0
    elif label == 'M':
        return 15
    elif label == 'C':
        return 20
    elif label == 'R':
        return 40
    elif label == 'O':
        return 50
    elif label == 'L':
        return 60
    elif label == 'H':
        return 80


# To combine location labels together by change them however we want
def change_location_labels(label):
    if label == 'C':
        return 'W'
    elif label == 'M' or label == 'R':
        return 'L'
    else:
        return label


# This function draws the plot that visualize all the activities that a user did each day
def draw_plot(start, end, result):
    result.drop(result.tail(1).index, inplace=True)
    mydates = pd.date_range(start, end).tolist()

    # This is the list of list images (length 48 each)
    complete_location = [[result['color'].iloc[i] for i in range(j * 48, (j + 1) * 48)] for j in
                         range(len(mydates) - 1)]

    empty_day = [0] * 48
    without_N_location = [ele for ele in complete_location if ele != empty_day]

    # This is the code where I draw the plots
    colors = ['navy', 'green', 'red', 'cyan', 'yellow', 'purple', 'magenta']
    cmap = mpl.colors.ListedColormap(colors)
    fig, ax = plt.subplots()
    extent = [0, 24, len(mydates), 0]
    ax.imshow(complete_location, extent=extent, aspect='auto', interpolation='none', cmap=cmap)
    plt.ylabel("Day")
    plt.xlabel("Time of day")
    plt.figure(figsize=(3, 3))
    plt.show()

    extent = [0, 24, len(without_N_location), 0]
    plt.imshow(without_N_location, extent=extent, aspect='auto', interpolation='none', cmap=cmap)
    plt.ylabel("Day")
    plt.xlabel("Time of day")
    plt.figure(figsize=(3, 3))
    plt.show()

    norec_patch = mpatches.Patch(color='navy', label='NoRec (N)')
    college_patch = mpatches.Patch(color='green', label='College (C)')
    market_patch = mpatches.Patch(color='red', label='Market (M)')
    road_patch = mpatches.Patch(color='cyan', label='Road (R)')
    other_patch = mpatches.Patch(color='yellow', label='Other (O)')
    leisure_patch = mpatches.Patch(color='purple', label='Leisure (L)')
    home_patch = mpatches.Patch(color='magenta', label='Home (H)')

    plt.legend(handles=[college_patch, home_patch, leisure_patch, market_patch, road_patch, other_patch, norec_patch])
    plt.show()


def fine_grain_location(location_data):
    df = {'Intervals': location_data['Intervals'], 'Label': location_data['Label']}
    df = pd.DataFrame(df)

    df['Label'] = df[['Intervals', 'Label']].groupby(['Intervals'])['Label'].transform(lambda x: ''.join(x))
    df.drop_duplicates(keep='last', inplace=True)
    df = df.reset_index()

    # We make sure that the date has 30 min. intervals
    df['date'] = pd.to_datetime(df['Intervals'])
    start = df['date'].iloc[0].strftime("%Y-%m-%d")
    s = df['date'].iloc[-1].strftime("%Y-%m-%d")
    date = datetime.strptime(s, "%Y-%m-%d")
    modified_end = date + timedelta(days=1)
    end = datetime.strftime(modified_end, "%Y-%m-%d")
    dff = pd.date_range(start, end, freq='30T')
    dff = pd.DataFrame(dff)
    dff['date'] = dff
    result = pd.merge(left=df, right=dff, how='right', left_on='date', right_on='date')

    # Here, we fill in the NAN labels to the desired 'N'
    result['Label'] = result['Label'].fillna('N')

    # Find the highest occurrence of the label
    result['Highest_Occ'] = ''
    result['Highest_Occ'] = [max(Counter(x), key=Counter(x).get) for x in result['Label']]

    # Remove and rename columns
    result = result.drop(columns=['Label', 'index', 'Intervals'], axis=1)

    # Add information for colour to visualizing it
    result['color'] = list(map(determine_color, result['Highest_Occ']))

    # For experimental purpose only:
    # this is the place where I combined the location labels together to compare different results for clustering.
    # In total: 4 labels
    # for i in range(len(result)):
    #     result['Highest_Occ'].iloc[i] = change_location_labels(result['Highest_Occ'].iloc[i])

    # Write them into a file to store it
    write_to_file(result)
    result = result.set_index('date')

    # Make a plot
    draw_plot(start, end, result)

    return result
