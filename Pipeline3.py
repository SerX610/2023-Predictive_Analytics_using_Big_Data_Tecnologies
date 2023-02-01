"""
.. module:: Pipeline3

Pipeline3
******

:Description: Pipeline3

    The execution of this script corresponds to the RUN-TIME CLASSIFIER PIPELINE
    Using the trained model in Pipeline 2, when an aircraft and a date is given,
    it predicts whether an unscheduled maitenance would happen during the
    following 7 days (last one included).
    When classifying, the KPIs (flighthours, flightcycles and delayedminutes)
    and the sensor 3453 value are needed. The code looks for these information
    in the avaibable sources (CSV directory and DW database). If it cannot find
    one or neither of these, no prediction is made because of lack of
    information.

:Authors:
    Sergio Cárdenas Gracia
    Aina Luis Vidal

:Date:  10/01/2023
"""

# GENERAL IMPORTS
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row
import os
from datetime import datetime

# IMPORTS NEEDED TO USE THE TRAINED MODEL
from pyspark.ml import Pipeline
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def read_new_record(n):
    """
    Determines whether a new record can be read to make a new prediction.
    """
    try:
        n[0] = str(input("Introduce the aircraft registration:"))
        return True
    except:
        return False

def extraction_sensor_values(sc, spark, input_aircraft, input_date):
    """
    Given a spark context, a spark session, an aircraft and a date, it reads all
    the csv files that corresponds to this aircraft and it returns a dataframe
    where each row corresponds to a flight done by the aircraft on the given
    date. It also returns a boolean variables that says whether the information
    of any flight is found.
    """
    info_found = False
    # CREATION OF AN EMPTY DATAFRAME WHICH WE WOULD FILL
    schema = StructType([
                  StructField('value', StringType(), True),
                  StructField('timeid', DateType(), True),
                  StructField('aircraftid', StringType(), True)
                  ])
    emptyRDD = spark.sparkContext.emptyRDD()
    df = spark.createDataFrame(emptyRDD, schema)

    # LECTURE OF THE CSV FILES
    list_files = os.listdir('resources/Data')
    for file in list_files:
        aircraft = file[-10: -4]
        id2 = datetime.strptime(input_date, '%Y-%m-%d')
        id3 = id2.date()
        # Read just the files for the input aircraft
        if aircraft == input_aircraft:
            data_file = spark.read.csv('resources/Data/'+file, sep=";", header=True)
            data_file = data_file.withColumn('date', data_file.date.cast('date'))
            # Select only the rows that corresponds to the input date
            data_file = data_file.select('value', 'date').filter(data_file['date'] == id3)
            if not(data_file.isEmpty()):
                info_found = True
                values = data_file
                values = values.withColumn('aircraftid', lit(aircraft))
                df = df.union(values)

    return df, info_found


def make_prediction(savedModel, record):
    """
    Given the trained model and a record, it returns the prediction of this
    record. It can be 1 for predicted unscheduled maitenance or 0 for no
    maintenance.
    """
    # ADAPTATION OF THE FORMAT
    assembler = VectorAssembler(inputCols=(["flighthours", "flightcycles", "delayedminutes", "sensorvalue"]), outputCol=("indexedFeatures"))
    data = assembler.transform(record)
    data = data.select("indexedFeatures")

    # MAKE THE PREDICTION
    predictions = savedModel.transform(data)
    prediction = predictions.select("prediction").collect()[0][0]
    return prediction


def pipeline3(sc, spark):
    """
    RUN-TIME CLASSIFIER PIPELINE
    It makes predictions at run-time. Once a new aircraft registration and a
    date are given, it extracts the corresponding sensor value and KPIs and it
    makes the prediction using the model trained in Pipeline 2.
    """

    # UPLOADS THE TRAINED MODEL (pipeline 2)
    savedModel = DecisionTreeClassificationModel.load('model')

    # CONNEXION TO THE DW DATABASE
    # We need the data in this table as for each entered aircraft and date, the
    # kpis values are needed to make the classification
    KPIs = (spark.read.format("jdbc").option("driver", "org.postgresql.Driver") \
                .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
                .option("dbtable", "public.aircraftutilization")\
                .option("user", "aina.luis")\
                .option("password", "DB150402")\
                .load())

    n = [0]
    # DEFINITION OF THE RUN-TIME CLASSIFIER
    # New entries would be readed since the execution of the program is cut
    while read_new_record(n):
        input_aircraft = n[0]
        input_date = input("Introduce the day (yyyy-mm-dd): ")

        # EXTRACTION OF THE SENSOR VALUE FOR THE ENTERED AIRCRAFT AND DAY
        df, info_found = extraction_sensor_values(sc, spark, input_aircraft, input_date)

        if not info_found:
            print("There is no avaibable information about aircraft", input_aircraft, "on date", input_date)
        else:
            # GROUP BY aircraftid AND timeid
            # We just want one row, correspoding to the aircraft and the date
            # given in the input.
            df = df.select('aircraftid', 'timeid', 'value')
            df = df.withColumn('value', df.value.cast('double'))
            df = df.withColumn('timeid', df.timeid.cast('date'))
            results = df.groupby('aircraftid', 'timeid').agg(avg('value').alias('sensorvalue'))

            # EXTRACTION OF THE KPIS VALUES FOR THE ENTERED AIRCRAFT AND DAY
            kpis = KPIs.select("aircraftid", "timeid", "flighthours", "flightcycles", "delayedminutes").filter((KPIs['aircraftid'] == input_aircraft) & (KPIs['timeid'] == input_date))
            record = results.join(kpis, ['aircraftid', 'timeid'], "inner")

            # SELECTION OF COLUMNS FOR ENTERING THE MODEL
            record = record.select("flighthours", "flightcycles", "delayedminutes", "sensorvalue")

            if record.isEmpty():
                print("There is no avaibable information about aircraft", input_aircraft, "on date", input_date)
            else:
                # MAKE THE PREDICTION
                prediction = make_prediction(savedModel, record)
                if prediction == 0:
                    print("No maintenance is expected for aircraft", input_aircraft, "on the 7 following days after", input_date)
                else:
                    print("An unscheduled maintenance is expected for aircraft", input_aircraft, "on the 7 following days after", input_date)
