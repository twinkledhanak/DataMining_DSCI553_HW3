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
sc = SparkContext('local[*]','HW3_Task1')

# spark = SparkConf().setAppName("HW3_Task1").setMaster("local[*]").set("spark.driver.bindAddress", "127.0.0.1")
# sc = SparkContext(spark)
# Change logger level to print only ERROR logs
sc.setLogLevel("ERROR")


# Calculate Jaccard Sim
def apply_jaccard_sim(x1,x2):
    # print("JS: x1: ",x1)
    # print("JS: x2: ",x2)
    # print("Matrix: ",matrix)
    intersection = set(matrix[x1])&set(matrix[x2])
    union = set(matrix[x1])|set(matrix[x2])
    jsim_val = len(intersection)/len(union)
    return x1,x2,jsim_val


def apply_local_hash(x):
    #print("Inside local_hash for: ",x)
    data = []#print("Inside local_hash for: ",x)
    n= int(num_rows/num_bands)
    for b in range(num_bands):
        #print("x[0]: ",x[0]," x[1]: ",x[1]," y: ",x[1][b * n:(b + 1) * n]," tuple : ",tuple(x[1][b * n:(b + 1) * n]))
        #           (( 24,   (0, 0) ) ,                   [‘82YGtjc5KKikNiqBZ33qzw’]  )
        data.append(((b, tuple(x[1][b * n:(b + 1) * n])), [x[0]]))
    #print("Data has been apended: ",data)    
    return data

## Hash function to be applied to every row of our matrix we generated
# @TODO - Refractor this function
def apply_custom_hash(h,row,m):
    #print("Inside custom_hash for: h: ",h," ,m: ",m," ,row: ",row)
    hashed_values = []
    for row in row[1]:
        #print("Row: ",row)
        val = (h[0]*row+h[1]) % m # as per assignment
        hashed_values.append(val)
    #print("Size of hashed_values array: ",len(hashed_values))
    return min(hashed_values)

start = time.time()

# Read arguments from command line
args = sys.argv
input_file_path = str(args[1])
output_fle_path = str(args[2])

# Initialize common variables
num_bands = 40
num_rows = 80

# Create a hash table
hash_table = np.random.randint(low = 1 ,high=1000,size = (num_rows,2))

## 1. Create an RDD from given text file
dataRDD = sc.textFile(str(input_file_path))
# Remove the CSV file header 
file_header = dataRDD.first()
dataRDD = dataRDD.filter(lambda row : row != file_header) 

## 2. Now create a tempRDD from rest of the file
tempRDD = dataRDD.map(lambda x: x.split(','))

## 3. Create a signature matrix
distinct_user_list = sorted(tempRDD.map(lambda x:x[0]).distinct().collect()) 
#print("Distinct user list: ",len(distinct_user_list))
## Now we convert every user value to number
user_value_list = np.arange(0,len(distinct_user_list),1)
#print("User_value_list: ",user_value_list)

## Maintain a mapping of user-value to user-number, eg, user1 : [0]
user_dict = dict(zip(distinct_user_list,user_value_list))
#print("User Dict: ",user_dict)
matRDD = tempRDD.map(lambda row: (row[1], user_dict[row[0]])).groupByKey().map(lambda row: (row[0], list(row[1]))).sortBy(lambda row: row[0])
matrix = matRDD.collectAsMap()
#print("Matrix:: ",len(matrix))

## 4. Custom Hashing 
permutatedRDD = matRDD.map(lambda row: (row[0], [apply_custom_hash(h,row,len(user_dict)) for h in hash_table]))
#print("After applying permutations to matrix:: ",permutatedRDD.collect())

## 5. Local Hashing
intermediateRDD = permutatedRDD.flatMap(apply_local_hash).reduceByKey(lambda x1, x2: x1 + x2).filter(lambda x: len(x[1]) > 1).flatMap(lambda x: sorted(list(itertools.combinations(sorted(x[1]), 2)))).distinct()
#print("Intermediate RDD: ",intermediateRDD.collect())

finalRDD = intermediateRDD.map(lambda row: apply_jaccard_sim(row[0], row[1])).filter(lambda row: row[2] >= 0.5).collect()
finalRDD  = sorted(finalRDD,key=lambda x:(x[0],x[1]))
#print("FINAL OUTPUT: ",finalRDD)



## Writing output to a file 
with open(output_fle_path,'w') as outfile:
    outfile.write("business_id_1, business_id_2, similarity\n")
    for elem in finalRDD:
        record = str(elem[0]) + str(",") + str(elem[1]) + str(",") + str(elem[2])
        outfile.write(record+"\n")
        
end = time.time()

# Also print the final execution time 
print("Elapsed time: ",(end-start))