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
sc = SparkContext('local[*]','HW3_Task2.3')

# spark = SparkConf().setAppName("HW3_Task1").setMaster("local[*]").set("spark.driver.bindAddress", "127.0.0.1")
# sc = SparkContext(spark)
# Change logger level to print only ERROR logs
sc.setLogLevel("ERROR")

## Lookup user and business feature map
def features(row):
    li = list()
    li.extend(user_feature_map.get(row[0]))
    li.extend(business_feature_map.get(row[1]))
    return li

def price(di, k):
    if di:
        if k in di.keys():
            return int(di.get(k))
    return 0

## Convert to numpy array
def convertToNPArray(data,type):
    li = []
    if type ==1:
        lables = []
        for d in data:
            li.append(features(d))
            lables.append(d[2])
        return np.asarray(li),np.asarray(lables)
    else:
        for d in data:
            li.append(features(d))
        return np.array(li)

## Calculate pearson coefficient
def calculate_pearson_coeff(neigh,users,busi,avg_rating):
    master_business_list = []
    master_user_list = []
    b_ = businessMap.get(neigh)
    n = businessRatings.get(neigh)
    for u in users:
        if b_.get(u):
            master_business_list.append(busi.get(u))
            master_user_list.append(b_.get(u))

    if len(master_business_list)!=0:
        num,d_1,d_2 = 0,0,0
        for i in range(0,len(master_business_list)):
            num += (master_business_list[i] - avg_rating)* (master_user_list[i] - n)
            d_1+= (master_business_list[i] - avg_rating)**2
            d_2+= (master_user_list[i] - n)**2
        denom = (d_1*d_2)**1/2
        if denom==0 and num==0: p=1
        elif denom==0: p=-1
        else: p = num/denom
    else:
        p = float(avg_rating/n)
    return p

## Predict - @TODO - refractor name and desc
def predict(p,avg):
    p = sorted(p,key=lambda x:x[0],reverse=True)
    cuttoff = 40
    num,denom=0,0
    if len(p)!=0:
        n = min(cuttoff,len(p))
        for i in range(n):
            num += p[i][0]*p[i][1]
            denom+= abs(p[i][0])
        pred_ = num/denom
        return min(5.0,max(0.0,pred_))
    else:
        return avg


## Predict function - @TODO - refractor name and desc
def func_pred(data):
    user_data = data[0]
    business_data = data[1]

    if business_data in businessMap:
        r = businessRatings.get(business_data)
        if userMap.get(user_data) is None:
            return user_data, business_data, str(r)

        list_busi = list(userMap.get(user_data))

        if len(list_busi) !=0 or list_busi is not None:
            p =list()
            for i in list_busi:
                curr = businessMap.get(i).get(user_data)
                coeff = calculate_pearson_coeff(i,list(businessMap.get(business_data)),businessMap.get(business_data),r)
                if coeff>0:
                    if coeff>1:
                        coeff = 1/coeff
                    p.append((coeff,curr))
            predict_ = predict(p,(userRatings.get(user_data)+r)/2)
            return user_data,business_data,min(5.0,max(0.0,predict_))
        else:
            return user_data,business_data,str(businessRatings.get(business_data))
    else:
        if len(list(userMap.get(user_data)))==0:
            return user_data, business_data, "2.5"

        return user_data,business_data,str(userRatings.get(user_data))





start = time.time()

# Read arguments from command line
args = sys.argv
input_file_path = str(args[1])
val_file_path = str(args[2])
output_fle_path = str(args[3])


train_yelp_path = os.path.join(input_file_path, 'yelp_train.csv')

## Read training data and create RDD
trainRDD = sc.textFile(train_yelp_path)
header = trainRDD.first()
trainRDD = trainRDD.filter(lambda line: line != header).map(lambda x: x.split(','))

## Read Testing data and create RDD
testRDD = sc.textFile(val_file_path)
header = testRDD.first()
testRDD = testRDD.filter(lambda x: x != header)

## Read user and business data
user_path=os.path.join(input_file_path, 'user.json')
business_path = os.path.join(input_file_path, 'business.json')

## Create user map 
user = trainRDD.map(lambda x: ((x[0]), ((x[1]), float(x[2])))).groupByKey().sortByKey(True).mapValues(dict)
userMap = user.collectAsMap()  # user key
## Get User ratings
userRatings = trainRDD.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()  # userId <-> avg rating

## Create business map
business = trainRDD.map(lambda x: ((x[1]), ((x[0]), float(x[2])))).groupByKey().sortByKey(True).mapValues(dict)
businessMap = business.collectAsMap()  # business key
## Get business ratings
businessRatings = trainRDD.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()  # businessId <-> avg rating


## Creating a matrix to test
test_matrix = testRDD.map(lambda x: x.split(",")).sortBy(lambda x: ((x[0]), (x[1]))).persist()

## Applying prediction to entire test matrix
initialPred = test_matrix.map(func_pred).map(lambda x: ((x[0], x[1]), float(x[2])))

## Create a user map with features
user_feature_map = sc.textFile(user_path).map(json.loads).map(
    lambda x: ((x["user_id"]), (x["review_count"], x["useful"], x["fans"], x["average_stars"]))).collectAsMap()

## Create a business map with features
business_feature_map = sc.textFile(business_path).map(json.loads).map(
    lambda x: ((x['business_id']), (x['stars'], x['review_count'], price(x['attributes'], 'RestaurantsPriceRange2')))).collectAsMap()

## Formatting training data
training_data, training_labels = convertToNPArray(data=trainRDD.collect(),type=1)
## Training model
trained_data = xgb.DMatrix(training_data, label=training_labels)
params = {}
params['max-depth'] = 15
params['objective'] = 'reg:linear'
model = xgb.train(params, trained_data, 100)

## Fomratting testing data
val_test = testRDD.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1])).collect()
test_data = convertToNPArray(data=val_test,type=0)

## Predicting on test data
prediction = model.predict(xgb.DMatrix(test_data))


## Writing predictions to a temporary output file
pred_file = "temp_output.csv"
with open(pred_file,'w') as outfile:
    outfile.write("user_id, business_id, prediction\n")
    for i in range(0, len(prediction)):
        record = str(val_test[i][0]) + "," + str(val_test[i][1]) + "," + str(max(1, min(5, prediction[i])))
        outfile.write(record + "\n")
outfile.close()

## Reading predcitions from temp file
tempRDD = sc.textFile(pred_file)
header = tempRDD.first()
resultRDD = tempRDD.filter(lambda line: line != header).map(lambda x: x.split(','))
pred_m = resultRDD.map(lambda x: (((x[0]), (x[1])), float(x[2])))

alpha = 0.1
alpha_1 = 1-alpha
#tempPred = initialPred.join(pred_m)
finalPred1 = initialPred.map(lambda x: ((x[0]), float((x[1][0] * alpha + x[1][1] * alpha_1))))
finalPred2 = pred_m.map(lambda x: ((x[0]), float((x[1][0] * alpha + x[1][1] * alpha_1))))

op_list1 = finalPred1.collect()
op_list2 = finalPred2.collect()
with open(output_fle_path,'w') as outfile:
    outfile.write("user_id, business_id, prediction\n")
    for i in range(len(op_list1)):
        record = str(op_list1[i][0][0]) + "," + str(op_list1[i][0][1]) + "," + str(op_list1[i][1])
        outfile.write( record + "\n")
    for i in range(len(op_list2)):
        record = str(op_list2[i][0][0]) + "," + str(op_list2[i][0][1]) + "," + str(op_list2[i][1])
        outfile.write( record + "\n")    



end = time.time()
duration = end-start
print(duration)