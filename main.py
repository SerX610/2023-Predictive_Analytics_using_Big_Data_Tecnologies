"""
.. module:: main

main
******

:Description: main

    The execution of this script executed all the code needed for solving the
    task of this project: develop a complex analytical pipeline for predicting
    if an aircraft is going to interrupts its opeartion unexpectedly in the
    seven net days.
    It uses 3 different scripts, one for each pipeline.
    1) Data Management Pipeline. It generates a matrix where rows denote
       the information of an aircraft per day, and the columns refer to the
       flighthours, flightcycles, delayedminutes KPIs and the label determining
       if an aircraft with these features would either have an unscheduled
       maitenance or no maitenance predicted in the next 7 days for that flight.

    2) Data Analysis Pipeline. Using the created matrix in Pipeline 1, it uses
       the MLlib library to train a decision tree. To do so, it creates two
       datasets (trainig and validation), it formats them according to what is
       expected by the MLlib library, it trains the model and it computes some
       traditional evaluation metrics (confusion matrix, accuracy, precision,
       recall).

    3) Run-time Classifier Pipeline. Given an aircraft and a day, it extracts
       the KPIs values from the DW and computes the average measurement of the
       sensor using the CSVs and it uses the model trained in Pipeline 2 to
       classify the record between maitenance or no maitenance.

    It has de possibility to execute either the three pipelines together or just
    one of them. In this last case, it is assumed that the execution of the
    previous pipelines has been done before as there is so depency between them
    (for executing Pipeline2, the matrix resulting from Pipeline1 is needed and
    for executing Pipeline3, the model resulting from Pipeline2 is needed).
     
:Authors:
    Sergio Cárdenas Gracia
    Aina Luis Vidal

:Date:  10/01/2023
"""

# GENERAL IMPORTS
import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

# IMPORT OF THE PIPELINES. They are defined in separated scripts
import Pipeline1
import Pipeline2
import Pipeline3

# DEFINITION OF GLOBAL VARIABLES
HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.6"
PYSPARK_DRIVER_PYTHON = "python3.6"


if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    # create the configuration
    conf = SparkConf()
    conf.set("spark.jars", JDBC_JAR)

    # create the spark session
    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    # create the spark context
    sc = pyspark.SparkContext.getOrCreate()

    if(len(sys.argv) < 2):
        # no pipeline specified --> execution of the whole process
        Pipeline1.pipeline1(sc, spark)
        Pipeline2.pipeline2(sc, spark)
        Pipeline3.pipeline3(sc, spark)
    elif(len(sys.argv) == 2):
        # a pipeline specified --> execution only of the specified one
        if(sys.argv[1] == "pipeline1"):
            Pipeline1.pipeline1(sc, spark)
        elif(sys.argv[1] == "pipeline2"):
            Pipeline2.pipeline2(sc, spark)
        elif(sys.argv[1] == "pipeline3"):
            Pipeline3.pipeline3(sc, spark)
        else:
           print("No  pipeline")
    else:
        print("Error in the input")
