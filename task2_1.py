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

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

# SparkContext(Master with number of partitions ,AppName)
sc = SparkContext('local[*]','HW3_Task2_1')

# spark = SparkConf().setAppName("HW3_Task1").setMaster("local[*]").set("spark.driver.bindAddress", "127.0.0.1")
# sc = SparkContext(spark)
# Change logger level to print only ERROR logs
sc.setLogLevel("ERROR")


## Function to calculate Pearson coefficient
def calc_pearson_coeff(neighbour,users,buss,avg_rating):
    master_business_list = []
    master_user_list = []
    b_ = business.get(neighbour)
    n = businessRating.get(neighbour)
    for u in users:
        if b_.get(u):
            master_business_list.append(buss.get(u))
            master_user_list.append(b_.get(u))

    if len(master_business_list)!=0:
        num, d_1, d_2 = 0,0,0
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


def predict(p,avg):
    p = sorted(p,key=lambda x:x[0],reverse=True)
    cutoff = 40
    num,denom=0,0
    if len(p)!=0:
        n = min(cutoff,len(p))
        for i in range(n):
            num += p[i][0]*p[i][1]
            denom+= abs(p[i][0])
        pred_val = num/denom
        return min(5.0,max(0.0,pred_val))
    else:
        return avg


######
def func_pred(data):
    #print("Data in func_pred: ",data)
    u = data[0]
    b = data[1] 
    if b in business:
        r = businessRating.get(b)
        #print("rating from business: ",r)
        if user.get(u) is None:
            return u, b, str(r)
        businessList = list(user.get(u))
        if len(businessList) !=0 or businessList is not None:
            p =list()
            for i in businessList:
                curr = business.get(i).get(u)
                coeff = calc_pearson_coeff(i,list(business.get(data[1])),business.get(data[1]),r)
                if coeff>0:
                    if coeff>1:
                        coeff = 1/coeff
                    p.append((coeff,curr))
            predict_result = predict(p,(userRating.get(u)+r)/2)
            return u,b,min(5.0,max(0.0,predict_result))
        else:
            return u,b,str(businessRating.get(b))
    else:
        if len(list(user.get(u)))==0 or list(user.get(u)) is None:
            return u, b, "2.5"

        return u,b,str(userRating.get(u))


start = time.time()


# Read arguments from command line
args = sys.argv
input_file_path = str(args[1])
val_file_path = str(args[2])
output_fle_path = str(args[3])

## 1. Create an RDD from given text file
dataRDD = sc.textFile(str(input_file_path))
# Remove the CSV file header 
file_header = dataRDD.first()
dataRDD = dataRDD.filter(lambda row : row != file_header) 

## 2. Now create a tempRDD from rest of the file
tempRDD = dataRDD.map(lambda x: x.split(','))

## 3. Now get all users from file
userDict = tempRDD.map(lambda row: ((row[0]), (row[1],float(row[2])))).groupByKey().sortByKey(True).mapValues(dict)
user = userDict.collectAsMap()
#print("Users from tempRDD: ",user)

## 4. Now get all businesses from file
businessDict = tempRDD.map(lambda row: ((row[1]), (row[0],float(row[2])))).groupByKey().sortByKey(True).mapValues(dict)
business = businessDict.collectAsMap()
#print("Business from tempRDD: ",business)

## 5. Now get all User ratings
userRating = tempRDD.map(lambda row:((row[0]),float(row[2]))).groupByKey().mapValues(lambda y : sum(y)/len(y)).collectAsMap()
#print("User ratings from tempRDD: ",userRating)

## 6. Now get all Business ratings
businessRating = tempRDD.map(lambda row:((row[1]),float(row[2]))).groupByKey().mapValues(lambda row:sum(row)/len(row)).collectAsMap()
#print("Business ratings from tempRDD: ",businessRating)

#######################################################
# Repeating same steps with Validation file again

## 7. Create an RDD from given text file
validationRDD = sc.textFile(str(val_file_path))
# Remove the CSV file header 
file_header = validationRDD.first()
validationRDD = validationRDD.filter(lambda row : row != file_header) 

## 8. Now create a tempRDD from rest of the file
valRDD = validationRDD.map(lambda x: x.split(','))

## 
finalValList = valRDD.map(func_pred).collect()
#print("Final Val List: ",finalValList)


## Writing output to a file 
with open(output_fle_path,'w') as outfile:
    outfile.write("user_id, business_id, prediction\n")
    for l in range(len(finalValList)):
        record = str(finalValList[l][0])+"," + str(finalValList[l][1])+"," + str(finalValList[l][2])
        outfile.write(record+"\n")

end = time.time()
# Also print the final execution time 
print("Elapsed time: ",(end-start))