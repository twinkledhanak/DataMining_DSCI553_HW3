# SparkContext is entrypoint to Spark, to connect to Spark clusters and create Spark RDD
from pyspark import SparkContext , SparkConf

import json 

## For memory and time limits
import sys
from resource import *
import time
import psutil
import os 
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb


os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

# SparkContext(Master with number of partitions ,AppName)
sc = SparkContext('local[*]','HW3_Task2.2')

# spark = SparkConf().setAppName("HW3_Task1").setMaster("local[*]").set("spark.driver.bindAddress", "127.0.0.1")
# sc = SparkContext(spark)
# Change logger level to print only ERROR logs
sc.setLogLevel("ERROR")

## Some common labels
yelp_train = "yelp_train.csv"
user_feature_file = "user.json"
business_feature_file = "business.json"

## Custom label encoder
def customLabelEncoder(arr):
    """
    Encode the unique values in `arr` as integers from 0 to n-1,
    where n is the number of unique values.

    Parameters:
        arr (numpy.ndarray): The array to encode.

    Returns:
        numpy.ndarray: The encoded array.
    """
    unique_vals, inverse = np.unique(arr, return_inverse=True)
    return inverse


## encoding string data before giving input to model
def encodeDataLabels(data):
    for c in data.columns:
        if data[c].dtype == 'object':
            #print("data: ",np.array(data[c].values))
            data[c] = customLabelEncoder(np.array(data[c].values))
    return data


## Custom training file
def createTrainData(train_data, userMap, businessMap):
    count = list()
    use = list()
    fan = list()
    rating = list()

    for u in train_data["user_id"]:
        #print("Train data user: ",u)
        if u in userMap.keys():
            #print("user from map: ",userMap.get(u)[0])
            count.append(userMap.get(u)[0])
            use.append(userMap.get(u)[1])
            fan.append(userMap.get(u)[2])
            rating.append(userMap.get(u)[3])
        else:
            # user is not present in yelp training file
            count.append(userCount)
            use.append(0)
            fan.append(0)
            rating.append(0)

    #print("size of count is: ",len(count))
    train_data['user_count'] = count
    train_data['user_useful'] = use
    train_data['fans'] = fan
    train_data['user_rating'] = rating



    count = list()
    use = list()
    rating = list()
    for b in train_data["business_id"]:
        #print("Checking business: ",b)
        if b in businessMap.keys():
            #print("business from map :: ",businessMap.get(b)[1])
            count.append(businessMap.get(b)[1])
            use.append(businessMap.get(b)[2])
            rating.append(businessMap.get(b)[0])
        else:
            ## current business is not present in yelp training file
            count.append(businessCount)
            use.append(2)
            rating.append(businessRating)
    train_data['business_count'] = count
    train_data['business_range'] = use
    train_data['business_rating'] = rating

    #print("Final training data is::  ",train_data)
    return train_data

## Function to get price 
def getPrice(dictionary, k):
    if dictionary:
        if k in dictionary.keys():
            return int(dictionary.get(k))
    return 0

start = time.time()

# Read arguments from command line
args = sys.argv
input_file_path = str(args[1])
val_file_path = str(args[2])
output_fle_path = str(args[3])

# Append paths for training and features 
yelp_train_path = os.path.join(input_file_path, yelp_train)
user_json_path = os.path.join(input_file_path, user_feature_file)
business_json_path = os.path.join(input_file_path, business_feature_file)

# Read the features for user file
userRDD = sc.textFile(user_json_path).map(json.loads).map(lambda row: ((row["user_id"]), (row["review_count"], row["useful"], row["fans"], row["average_stars"]))).persist()
userMap = userRDD.collectAsMap()

# Read the features for business file
businessRDD = sc.textFile(business_json_path).map(json.loads).map(lambda row: (
    (row['business_id']), (row['stars'], row['review_count'], getPrice(row['attributes'], 'RestaurantsPriceRange2')))).persist()
businessMap = businessRDD.collectAsMap()

## Get average user ratings and counts
userRating = userRDD.map(lambda x: x[1][3]).mean()
userCount = userRDD.map(lambda x: x[1][0]).mean()

## Get avergae business ratings and counts
businessRating = businessRDD.map(lambda x: x[1][0]).mean()
businessCount = businessRDD.map(lambda x: x[1][1]).mean()

## Now reading the training data and validation data
train_data = pd.read_csv(yelp_train_path)
## Now we create complete training data using other user/business feature maps
train_data = createTrainData(train_data, userMap, businessMap)

## Encoding string data to numeric format
## [455854 rows x 10 columns] 
train_data = encodeDataLabels(train_data)

## Create final train data
Xtrain = train_data.drop(["stars"], axis=1)
Ytrain = train_data.stars.values

## Training model
model = xgb.XGBRegressor(learning_rate=0.3) # @TODO - experiment with different lr values
model.fit(Xtrain, Ytrain)

## Same steps as Train data for Test data
data_testing = pd.read_csv(val_file_path)
test_data = createTrainData(data_testing.copy(), userMap, businessMap)
test_data = encodeDataLabels(test_data)

Xtest = test_data.drop(["stars"], axis=1)
ypred = model.predict(data=Xtest)

#print("Ypred: ",ypred)
## Writing output to a file 
with open(output_fle_path,'w') as outfile:
    outfile.write("user_id, business_id, prediction\n")
    for i in range(0,len(ypred)):
        record = str(data_testing.user_id[i]) + "," + str(data_testing.business_id[i]) + "," + str(ypred[i])
        #print(ypred[i], " | " ,data_testing.user_id[i] , " | ", data_testing.business_id[i])
        outfile.write(record+"\n")
        

end = time.time()
# Also print the final execution time 
print("Elapsed time: ",(end-start))