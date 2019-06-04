#Keras model with the same architecture as that in "Make You Own Neural Network" by T. Rashid
import tensorflow as tf
import numpy as np
#to measure elapsed time
from timeit import default_timer as timer
import datetime
#pandas for reading CSV files
import pandas as pd
from pandas.api.types import CategoricalDtype
import os
import keras
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import sys
import csv
import heapq

#np.set_printoptions(threshold=sys.maxsize)
#setting for dynamic memory allocation in GPU to avoid CUBLAS_STATUS_ALLOC_FAILED error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def simplifytoBoolean(row, strings, col):
    if any(x in row[col] for x in strings):
        boolVal = "Yes"
    elif row[col] == "Unknown":
        boolVal = "Unknown"
    else:
        boolVal = "No"
    return boolVal

def secondaryColor(row):
    identifier = "/"
    if identifier in row["Color"]:
        secondaryColor = row["Color"].split(identifier,1)[1]
    else:
        secondaryColor = "None"
    return secondaryColor

def convertAge(row):
    cell = row["AgeuponOutcome"]
    if not "Unknown" in cell:
        age = int(cell.split(" ", 1)[0])

    returnval = cell

    if "0" in cell or "Unknown" in cell:
        returnval = 0 
    elif "years" in cell or "year" in cell:
        ageInDays = age*365
        returnval = ageInDays
    elif "months" in cell or "month" in cell:
        ageInDays = age*30
        returnval = ageInDays
    elif "weeks" in cell or "week" in cell:
        ageInDays = age*7
        returnval = ageInDays
    elif "day" in cell or "days" in cell: #to make results uniform
        returnval = age

    return returnval

def convertDateTime(row, conversion):
    cell = row["DateTime"]
    date = datetime.datetime.strptime(cell, "%m/%d/%Y %H:%M")

    if conversion == "Month":
        convertedUnit = date.month
    elif conversion == "Day":
        convertedUnit = date.strftime('%A')
    elif conversion == "Hour":
        convertedUnit = date.hour
    else:
        print("Unexpected conversion in convertDateTime function")
        exit(1)

    return convertedUnit

def categoricalToNumeric(df):
    #df = pd.get_dummies(df, columns=["AnimalType", "SexuponOutcome", "Breed", "PrimaryColor", "SecondaryColor", "IsMix", "N/S", "DayofWeek"])
    df = pd.get_dummies(df, columns=["AnimalType", "SexuponOutcome", "Breed", "PrimaryColor", "IsMix", "N/S", "DayofWeek"])

    #Re-arrange dataframe to put OutcomeType as the last column
    if 'OutcomeType' in df.columns:
        cols = list(df)
        cols.insert(len(cols), cols.pop(cols.index('OutcomeType')))
        df = df.ix[:, cols]
    return df

def dataManipulation(df):
    #df = df.drop("AnimalID", axis=1) 
    df = df.drop("Name", axis=1) #don't think name or outcomesubtype are relevant, so removing from dataframe for now
    if 'OutcomeSubtype' in df.columns:
        df = df.drop("OutcomeSubtype", axis=1)
    if 'AnimalID' in df.columns:
        df.rename(columns={'AnimalID' : 'ID'}, inplace=True)

    #df["SecondaryColor"] = df.apply(secondaryColor, axis=1) #creating column with the secondary color of the animal

    num1 = df["Color"].nunique() #just to see how many unique values
    df["Color"] = df["Color"].replace("\/.*", "", regex=True) #Cutting everything after / in Color column
    df.rename(columns={'Color' : 'PrimaryColor'}, inplace=True)
    num2 = df["PrimaryColor"].nunique()
    df["IsMix"] = df.apply(simplifytoBoolean, args=(["Mix", "/"], "Breed"), axis=1) #Creating column that states whether animal is a mix or not

    num3 = df["Breed"].nunique()

    df["Breed"] = df["Breed"].replace(" Mix", "/", regex=True) #Cutting everything after / or space in Breed column
    df["Breed"] = df["Breed"].replace("\/.*", "", regex=True)

    num4 = df["Breed"].nunique()

    df.loc[df["SexuponOutcome"].isnull(),"SexuponOutcome"] = "Unknown" #fill empty cells with value "unknown"
    df.loc[df["AgeuponOutcome"].isnull(),"AgeuponOutcome"] = "Unknown" #fill empty cells with value "unknown"

    #Creating column that states whether the animal is neutered/spayed
    df["N/S"] = df.apply(simplifytoBoolean, args=(["Spayed", "Neutered"], "SexuponOutcome"), axis=1) #N/S is shortened form for Neutered/Spayed

    #Cut everything other than Male/Female from SexuponOutcome column
    df["SexuponOutcome"] = df["SexuponOutcome"].replace(".* ", "", regex=True)

    #Convert all ages to same units (days) for AgeuponOutcome column
    df["AgeInDays"] = df.apply(convertAge, axis=1)
    df = df.drop("AgeuponOutcome", axis=1) #remove DateTime column now that we've extracted useful information 

    #Convert DateTime to Month Column, DayofWeek Column, and ApproxHour Column
    df["Month"] = df.apply(convertDateTime, args=("Month",), axis=1)
    df["DayofWeek"] = df.apply(convertDateTime, args=("Day",), axis=1)
    df["ApproxHour"] = df.apply(convertDateTime, args=("Hour",), axis=1)
    df = df.drop("DateTime", axis=1) #remove DateTime column now that we've extracted useful information 

    return df
data = pd.read_csv("train_90.csv") # store csv into dataframe (data)
data = dataManipulation(data)

test_data = pd.read_csv("test_10.csv")
test_data = dataManipulation(test_data)
#test_data["OutcomeType"] = 'None'
all_data = pd.concat([data, test_data])

#to ensure that training and test data have same number of inputs, due to different breeds, colors, etc. being present in one set and not the other
for col in all_data.select_dtypes(include=[np.object]).columns:
    unique_values = all_data[col].dropna().unique()
    data[col] = data[col].astype(CategoricalDtype(categories=unique_values))
    test_data[col] = test_data[col].astype(CategoricalDtype(categories=unique_values))

data = categoricalToNumeric(data)
test_data = categoricalToNumeric(test_data)
model = load_model('final_model.h5')
input_nodes = len(data.columns) - 2
outputs_array = np.zeros((len(test_data.index), 5))

output_labels = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']

correct_labels = []
gotten_labels = []
for index, row in test_data.iterrows():
    correct_label = row['OutcomeType']
    correct_labels.append(correct_label)
    inputs = (row[1:input_nodes+1]).values
    outputs = model.predict(np.reshape(inputs, (1, len(inputs)))) #numpy ndarray type
    label_index = np.argmax(outputs)
    outputs_array[index,:] = outputs    
    gotten_labels.append(output_labels[label_index])

results = pd.DataFrame(outputs_array, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
results.insert(0, "ID", test_data["ID"])

label_names = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']

cm = confusion_matrix(correct_labels, gotten_labels)

plot_confusion_matrix(cm, target_names=label_names, title='Confusion Matrix')

results.to_csv(r'C:\Users\Khakh\Desktop\Spring 2019\ENSC413\Project\Final\results.csv', index=None)