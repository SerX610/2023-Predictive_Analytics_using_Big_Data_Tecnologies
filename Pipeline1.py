"""
.. module:: Pipeline1

Pipeline1
******

:Description: Pipeline1

    The execution of this script corresponds to the DATA MANAGEMENT PIPELINE.
    It generates a matrix where rows denote the information of an aircraft per
    day, and the columns refer to the flighthours, flightcycles, delayedminutes
    KPIs and the label determining whether this aircraft have an unscheduled
    maitenance or not in the next 7 days for that day (0 denotes no maintenance
    and 1 denotes the opposite).
    For doing so, it extracts the sensor values from the CSV files, then joins
    these information with the KPIs values extracted from the DW database and
    then labels each row considering the data in the AMOS database. 

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

# The pandas package is only used for saving the resulting matrix in a csv file.
# The whole code for the Data Management Pipeline is done using pyspark
# dataframes, not the pandas ones.
import pandas


def extraction_sensor_values(sc, spark):
    """
    Given a spark context and a spark session, it reads all the csv files
    contained in the resources/trainingData directory and returns a pyspark
    dataframe where each row has the average sensor value for an aircraft
    and a day.
    """
    # CREATION OF AN EMPTY DATAFRAME WHICH WE WOULD FILL
    schema = StructType([
                  StructField('value', StringType(), True),
                  StructField('timeid', TimestampType(), True),
                  StructField('aircraftid', StringType(), True)
                  ])
    emptyRDD = spark.sparkContext.emptyRDD()
    df = spark.createDataFrame(emptyRDD, schema)

    # LECTURE OF THE CSV FILES
    list_files = os.listdir('resources/trainingData')
    for file in list_files:
        data_file = spark.read.csv('resources/trainingData/'+file, sep=";", header=True)
        values = data_file.select('value', 'date')
        # Select aircraft and include a column with the aircraft registration fixed
        aircraft = file[-10: -4]
        values = values.withColumn('aircraftid', lit(aircraft))
        df = df.union(values)

    # GROUP BY aircraftid AND timeid
    # (we can have different rows for the same aircraft and the same day if an
    # aircraft does more than one flight in a day and we want them together with
    # the average sensor value)
    df = df.select('aircraftid', 'timeid', 'value')
    df = df.withColumn('value', df.value.cast('double'))
    df = df.withColumn('timeid', df.timeid.cast('date'))
    results = df.groupby('aircraftid', 'timeid').agg(avg('value').alias('sensorvalue'))
    return results


def extraction_kpis_values(sc, spark):
    """
    Given a spark context and a spark session, it makes a connection to the
    aircraftutilization table of the DW database and returns a dataframe where
    each row has the flighthours, flightcycles and delayedminutes KPI of an
    aircraft during a day.
    """
    # CONNEXION TO THE DW DATABASE
    KPIs = (spark.read.format("jdbc").option("driver", "org.postgresql.Driver") \
                .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require") \
                .option("dbtable", "public.aircraftutilization")\
                .option("user", "aina.luis")\
                .option("password", "DB150402")\
                .load())

    # SELECTION OF THE COLUMNS WE ARE INTERESTED IN
    kpis = KPIs.select("aircraftid", "timeid", "flighthours", "flightcycles", "delayedminutes")
    kpis = kpis.withColumn('timeid', kpis.timeid.cast('date'))
    return kpis

def dataframe_labeling(sc, spark, matrix):
    """
    Given a spark context, a spark session and a matrix with the KPIs and sensor
    values for a given an aircraft during a day, it makes a connexion to the
    operationinterruption table in AMOS database and returns the same matrix but
    with each row labelled.
    The labelling has two categories: 0 for no maitenance and 1 for unscheduled
    maitenance. An aircraft is considered to be part of category 1 if it exists
    an unscheduled operation interruption cased by sensor 3453 during the
    following 7 days (last one included) than the timeid date, registered in the
    AMOS table.
    """
    # CONNEXION TO THE AMOS DATABASE
    Operations = (spark.read.format("jdbc").option("driver", "org.postgresql.Driver") \
                .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require") \
                .option("dbtable", "oldinstance.operationinterruption")\
                .option("user", "aina.luis")\
                .option("password", "DB150402")\
                .load())

    # SELECTION OF THE COLUMNS AND ROWS WE ARE INTERESTED IN
    # A filter is applied because we are only interested in those rows
    # corresponding to unscheduled interruption operations (of kind Delay,
    # AircraftOnGround or Safety) that are caused by the sensor 3453.
    unscheduled = Operations.select('aircraftregistration', 'starttime').filter(((Operations['kind'] == 'Delay') | (Operations['kind'] == 'AircraftOnGround') | (Operations['kind'] == 'Safety')) & (Operations['subsystem'] == '3453'))
    unscheduled = unscheduled.withColumn('starttime', unscheduled.starttime.cast('date'))

    # LABELING THE DATAFRAME
    # A row would be marked with the label of maintenance (1) if it exists an
    # unscheduled operation interruption cased by sensor 3453 during the
    # following 7 days (included) than the timeid date.
    df_cross = matrix.crossJoin(unscheduled).where(matrix['aircraftid'] == unscheduled['aircraftregistration'])
    df_cross = df_cross.withColumn('difference', (df_cross.starttime-df_cross.timeid))
    df_cross = df_cross.withColumn('difference', df_cross.difference.cast('int'))
    df_cross = df_cross.withColumn('maintenance', when(((df_cross.difference <= 7) & (df_cross.difference >=0)), 1).otherwise(0))
    df_cross = df_cross.groupby('aircraftid', 'timeid', 'flighthours', 'flightcycles', 'delayedminutes', 'sensorvalue').agg(max('maintenance').alias('maintenance'))
    return df_cross


def pipeline1(sc, spark):
    """
    DATA MANAGEMENT PIPELINE
    It extracts both the sensor values from the CSV files and the KPIs values
    from the DW database and then it labels each row considering the unscheduled
    operation interruptions registered by the 3453 sensor in the AMOS database.
    """

    # EXTRATION OF THE DATA FROM THE SOURCES
    sensor_values = extraction_sensor_values(sc, spark)
    kpis = extraction_kpis_values(sc, spark)

    # JOIN THE KPIs VALUES FOR EACH AIRCRAFT - DAY WITH THE SENSOR VALUES
    matrix = sensor_values.join(kpis, ['aircraftid','timeid'], "inner")

    # LABEL EACH ROW WITH "unscheduled maintenance" (1) OR "no maintenance" (0)
    df_labelled = dataframe_labeling(sc, spark, matrix)

    # EXPORTATION OF THE RESULTING MATRIX
    data = df_labelled.select('maintenance', 'flighthours', 'flightcycles', 'delayedminutes', 'sensorvalue')
    data.toPandas().to_csv("data_def.csv", header=True, index=False)
