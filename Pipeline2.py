"""
.. module:: Pipeline2

Pipeline2
******

:Description: Pipeline2

    The execution of this script corresponds to the DATA ANALYSIS PIPELINE.
    Using the created matrix in Pipeline 1, it uses the MLlib library to train a
    decision tree. To do so, it splits the matrix into two datasets (training
    and validation), it formats them according to what is expected by the MLlib
    library, it trains the model and it computes some traditional evaluation
    metrics (confusion matrix, accuracy, precision, recall).

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
#import os

# IMPORTS NEEDED TO TRAIN THE MODEL
import random
from pyspark.ml import Pipeline
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

def compute_confusion_matrix(prediction_list):
    """
    Given the prediction list (list of tuples where each of them has the true
    label and the prediction) returns the confusion matrix.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, l in prediction_list:
        if p==1 and l == 1: tp += 1
        if p==1 and l == 0: fp += 1
        if p==0 and l == 1: fn += 1
        if p==0 and l == 0: tn += 1
    return [[tp, fp], [fn, tn]]

def compute_accuracy(confusion_matrix):
    """
    Given the confusion matrix, it returns the accuracy.
    """
    tp = confusion_matrix[0][0]
    tn = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    return (tp+tn)/(tp+tn+fp+fn)

def compute_recall(confusion_matrix):
    """
    Given the confusion matrix, it returns the recall metric for both the
    positive and the negative label.
    """
    tp = confusion_matrix[0][0]
    tn = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    return tp/(tp+fn), tn/(tn+fp)

def compute_precision(confusion_matrix):
    """
    Given the confusion matrix, it returns the precision metric for both the
    positive and the negative label.
    """
    tp = confusion_matrix[0][0]
    tn = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    return tp/(tp+fp), tn/(tn+fn)

def compute_evaluation_metrics(predictions):
    """
    Given a dataframe with the predictions of the model, it computes some
    traditional evaluation metrics. The confusion matrix, the accuracy, the
    recall positive, the recall negative, the precision positive and the
    precision negative are returned.
    """
    # ADAPTATION OF THE FORMAT OF THE PREDICTIONS
    predictionAndLabels = predictions.select("prediction", "indexedLabel").collect()
    prediction_list = [(row.prediction, row.indexedLabel) for row in predictionAndLabels]

    # COMPUTATION OF THE EVALUATION METRICS
    confusion_matrix = compute_confusion_matrix(prediction_list)
    accuracy = compute_accuracy(confusion_matrix)
    recall_positive, recall_negative = compute_recall(confusion_matrix)
    precision_positive, precision_negative = compute_precision(confusion_matrix)

    return accuracy, recall_positive, recall_negative, precision_positive, precision_negative, confusion_matrix


def pipeline2(sc, spark):
    """
    DATA ANALYSIS PIPELINE
    It uploads the matrix generated in Pipeline 1 and it uses the MLlib library
    to train a decision tree. The goal of the classifier is to predict whether
    an aircraft on a particular day would have an unscheduled maitenance in the
    following 7 days (last one included).
    """

    # UPLOAD THE MATRIX (pipeline 1)
    matrix = spark.read.csv('./data_def.csv', sep=",", header=True)
    matrix = matrix.withColumn('flighthours', matrix.flighthours.cast('double'))
    matrix = matrix.withColumn('flightcycles', matrix.flightcycles.cast('double'))
    matrix = matrix.withColumn('delayedminutes', matrix.delayedminutes.cast('double'))
    matrix = matrix.withColumn('sensorvalue', matrix.sensorvalue.cast('double'))

    # ADAPTATION OF THE FORMAT
    # For training the model using the MLlib library, we need to have two
    # columns, one for the label and another one with all the features used to
    # made the prediction. We index both columns with indexedLabel and
    # indexedFeatures, respectively.
    assembler = VectorAssembler(inputCols=(["flighthours", "flightcycles", "delayedminutes", "sensorvalue"]), outputCol=("features"))
    data = assembler.transform(matrix)
    data = data.select("features", "maintenance")
    random.seed(1504010)
    labelIndexer = StringIndexer(inputCol="maintenance", outputCol="indexedLabel").fit(data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # SPLIT THE DATA INTO TRAINING AND TEST SETS
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # TRAIN THE DECISION TREE
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    model = pipeline.fit(trainingData)

    # MAKE PREDICTIONS USING THE TEST SET
    predictions = model.transform(testData)

    # COMPUTATION OF THE EVALUATION METRICS
    accuracy, recall_positive, recall_negative, precision_positive, precision_negative, confusion_matrix = compute_evaluation_metrics(predictions)
    print("Accuracy = ", accuracy)
    print("Recall positive = ", recall_positive)
    print("Recall negative = ", recall_negative)
    print("Precision positive = ", precision_positive)
    print("Precision negative = ", precision_negative)
    print("Test Error = %g " % (1.0 - accuracy))
    print("Confusion matrix = ", confusion_matrix)

    # EXPORTATION OF THE RESULTING MODEL
    treeModel = model.stages[2]
    treeModel.write().overwrite().save('model')
