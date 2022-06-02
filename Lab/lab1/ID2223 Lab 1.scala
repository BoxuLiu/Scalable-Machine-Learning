// Databricks notebook source
import org.apache.spark.SparkConf
import org.apache.spark.ml._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.streaming._
import org.apache.spark.sql.types._
import org.apache.spark.sql.streaming.{GroupStateTimeout, OutputMode}
//import org.apache.hadoop.hbase.spark.datasources.HBaseTableCatalog
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

// COMMAND ----------

//1. Get the data
val housing = spark.read.format("csv").option("header","true").option("inferSchema", "true").option("sep",",").load("/FileStore/tables/data/housing.csv")
import spark.implicits._

// COMMAND ----------

//2.1. Schema and dimension
housing.printSchema

// COMMAND ----------

housing.count

// COMMAND ----------

//2.2. Look at the data
housing.show(5)

// COMMAND ----------

//Print the number of records with population more than 10000.
housing.filter(col("population") >= 10000)

// COMMAND ----------

//2.3. Statistical summary
//Print a summary of the table statistics for the attributes
housing.describe("housing_median_age").show
housing.describe("total_rooms").show
housing.describe("median_house_value").show
housing.describe("population").show

// COMMAND ----------

//maximum age, the minimum number of rooms,the average of house values
housing.select(max("housing_median_age"),min("total_rooms"),avg("median_house_value")).show

// COMMAND ----------

//2.4. Breakdown the data by categorical data
//Print the number of houses in different areas
housing.select(col("ocean_proximity")).withColumn("count",lit(1)).groupBy("ocean_proximity").sum().sort($"sum(count)".desc).show
//housing.select().distinct.show

// COMMAND ----------

//Print the average value of the houses
housing.select(col("ocean_proximity"), col("median_house_value").cast(DoubleType))
                .groupBy("ocean_proximity").avg()
                .withColumnRenamed("avg(median_house_value)","avg_value").show

// COMMAND ----------

//Rewrite the above question in SQL
housing.createOrReplaceTempView("df")

spark.sql("SELECT ocean_proximity, COUNT(ocean_proximity) FROM df GROUP BY ocean_proximity ORDER BY COUNT(ocean_proximity) DESC").show()
spark.sql("SELECT ocean_proximity, AVG(median_house_value) AS avg_value FROM df GROUP BY ocean_proximity").show()


// COMMAND ----------

//2.5Correlation among attributes
val va = new VectorAssembler().setInputCols(Array("housing_median_age","total_rooms","median_house_value","population")).setOutputCol("correlation")

val housingAttrs = va.transform(housing)

housingAttrs.show(5)


// COMMAND ----------

import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

val Row(coeff: Matrix) = Correlation.corr(housingAttrs, "correlation").head

println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

//2.6. Combine and make new attributes
val housingCol1 = housing.withColumn("rooms_per_household", col("total_rooms") / col("households"))
val housingCol2 = housingCol1.withColumn("bedrooms_per_room", col("total_bedrooms") / col("total_rooms"))
val housingExtra = housingCol2.withColumn("population_per_household",col("population") / col("households"))

housingExtra.select("rooms_per_household", "bedrooms_per_room", "population_per_household").show(5)


// COMMAND ----------

//3.Prepare the data for Machine Learning algorithms
//3.1. Prepare continuse attributes
//Data cleaning
val renamedHousing = housingExtra.withColumnRenamed("median_house_value","label")
// label columns
val colLabel = "label"

// categorical columns
val colCat = "ocean_proximity"

// numerical columns
val colNum = renamedHousing.columns.filter(_ != colLabel).filter(_ != colCat)

// COMMAND ----------

for (c <- colNum) {
    val count_missing_values = renamedHousing.filter(s"${c} is null").count()
    println(s"Number of missing values in column ${c}: ${count_missing_values}")
}

// COMMAND ----------

//completes missing values in a dataset, either using the mean or the median
import org.apache.spark.ml.feature.Imputer

val imputer = new Imputer().setStrategy("median").setInputCols(Array("total_bedrooms", "bedrooms_per_room")).setOutputCols(Array("total_bedrooms", "bedrooms_per_room"))                           
val imputedHousing = imputer.fit(renamedHousing).transform(renamedHousing)

imputedHousing.select("total_bedrooms", "bedrooms_per_room").show(5)

// COMMAND ----------

//Scaling
//standardization
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

val va = new VectorAssembler().setInputCols(colNum).setOutputCol("allFeatures")
val featuredHousing = va.transform(imputedHousing)

val scaler = new StandardScaler()
    .setInputCol("allFeatures")
    .setOutputCol("standerFeatures")
    .setWithStd(true)
    .setWithMean(true)
val scaledHousing = scaler.fit(featuredHousing).transform(featuredHousing)

scaledHousing.show(5)

// COMMAND ----------

//3.2. Prepare categorical attributes
renamedHousing.select(col("ocean_proximity")).distinct().show

// COMMAND ----------

//String indexer
import org.apache.spark.ml.feature.StringIndexer

val indexer = new StringIndexer().setInputCol("ocean_proximity").setOutputCol("ocean_proximityInd")
val idxHousing = indexer.fit(renamedHousing).transform(renamedHousing)

idxHousing.show(5)

// COMMAND ----------

indexer.fit(renamedHousing).labelsArray

// COMMAND ----------

//One-hot encoding
import org.apache.spark.ml.feature.OneHotEncoder

val encoder = new OneHotEncoder().setInputCol("ocean_proximityInd").setOutputCol("one-hot")
val ohHousing = encoder.fit(idxHousing).transform(idxHousing)

ohHousing.show(5)

// COMMAND ----------

//4. Pipeline
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}

val numPipeline = new Pipeline().setStages(Array(imputer,va,scaler))
val catPipeline = new Pipeline().setStages(Array(indexer,encoder))

val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newHousing = pipeline.fit(renamedHousing).transform(renamedHousing)

newHousing.show(5)

// COMMAND ----------

val va2 = new VectorAssembler().setInputCols(Array("standerFeatures","one-hot")).setOutputCol("features")
val dataset = va2.transform(newHousing).select("features", "label")

dataset.show(5)

// COMMAND ----------

val Array(trainSet, testSet) = dataset.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

//5. Make a model
import org.apache.spark.ml.regression.LinearRegression
//5.1. Linear regression model
// train the model
val lr = new LinearRegression().setMaxIter(15)
val lrModel = lr.fit(trainSet)
val trainingSummary = lrModel.summary

println(s"Coefficients: ${lrModel.coefficients}, Intercept: ${lrModel.intercept}")
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")

// COMMAND ----------


import org.apache.spark.ml.evaluation.RegressionEvaluator

// make predictions on the test data
val predictions = lrModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

//5.2. Decision tree regression
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val dt = new DecisionTreeRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")

// train the model
val dtModel = dt.fit(trainSet)

// make predictions on the test data
val predictions = dtModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

//5.3. Random forest regression
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val rf = new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")

// train the model
val rfModel = rf.fit(trainSet)

// make predictions on the test data
val predictions = rfModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

//5.4. Gradient-boosted tree regression
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val gb = new GBTRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")

// train the model
val gbModel = gb.fit(trainSet)

// make predictions on the test data
val predictions = gbModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

//6. Hyperparameter tuning by using Random Forest model
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator

val paramGrid = new ParamGridBuilder()
    .addGrid(rf.numTrees, Array(1,5,10))
    .addGrid(rf.maxDepth, Array(5,10,15))
    .build()

val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

val cv = new CrossValidator()
    .setEstimator(rf)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)

val cvModel = cv.fit(trainSet)

val predictions = cvModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

//7. An End-to-End Classification Test
val ccdefault = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("/FileStore/tables/data/ccdefault.csv")
ccdefuault.printSchema

// COMMAND ----------

//Load the data
ccdefault.count

// COMMAND ----------

ccdefault.show(5)

// COMMAND ----------

val renamedCC = ccdefault.withColumnRenamed("DEFAULT", "label").drop("ID")
val colLabel = "label"
val colNum = renamedCC.columns.filter(_ != colLabel)
for (c <- colNum) {
   println(s"Number of missing values in column $c:  ${renamedCC.filter(renamedCC(c).isNull).count()}")
}

// COMMAND ----------

import org.apache.spark.ml.feature.PCA
val va = new VectorAssembler().setInputCols(colNum)
                              .setOutputCol("allFeatures")
val ccFeatures = va.transform(renamedCC)

val pca = new PCA()
              .setInputCol("allFeatures")
              .setOutputCol("PCAFeatures")
              .setK(19)
val pcaDataSet = pca.fit(ccFeatures).transform(ccFeatures)

pcaDataSet.show

// COMMAND ----------

val scaler = new StandardScaler().setInputCol("PCAFeatures")
                                 .setOutputCol("standerFeatures")
                                 .setWithStd(true)
                                 .setWithMean(true)

val scaledCC = scaler.fit(pcaDataSet).transform(pcaDataSet)

scaledCC.show(5)

// COMMAND ----------

//form a pipeline
val numPipeline = new Pipeline()
  .setStages(Array(va, pca, scaler))

val ccPipeline = new Pipeline().setStages(Array(numPipeline))

val newCC = ccPipeline.fit(renamedCC).transform(renamedCC)

newCC.show(5)

// COMMAND ----------

val ccVa2 = new VectorAssembler().setInputCols(Array("standerFeatures"))
                               .setOutputCol("features")
val ccDataSet = ccVa2.transform(newCC).select("features", "label")

ccDataSet.show(5)

// COMMAND ----------

val Array(training, test) = ccDataSet.randomSplit(Array(0.8, 0.2))
training.show

// COMMAND ----------

//LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression

val ccLR = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")

val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("label")  
  .setRawPredictionCol("prediction")
  .setMetricName("areaUnderROC")
 
  

val paramGrid = new ParamGridBuilder()
  .addGrid(ccLR.maxIter,Array(15,100))
  .addGrid(ccLR.regParam,Array(0.3,0.5))
  .addGrid(ccLR.elasticNetParam,Array(0.3,0.5,0.8))
  .build()

val cv = new CrossValidator()
  .setEstimator(ccLR)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)
val cvModel = cv.fit(training)

val predictions = cvModel.transform(test)
predictions.select("prediction", "label", "features").show(10)

val roc = evaluator.evaluate(predictions)
println(s"ROC on test data = $roc")

// COMMAND ----------

import org.apache.spark.ml.classification.DecisionTreeClassifier

val ccDT = new  DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label")

val paramGrid = new ParamGridBuilder()
  .addGrid(dt.maxDepth,Array(5,10,15))
  .build()

val evaluator = new BinaryClassificationEvaluator()
  .setRawPredictionCol("prediction")
  .setLabelCol("label")
  .setMetricName("areaUnderROC")
  

val cv = new CrossValidator()
  .setEstimator(ccDT)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

val cvModel = cv.fit(training)

val predictions = cvModel.transform(test)
predictions.select("prediction", "label", "features").show(5)

val roc = evaluator.evaluate(predictions)
println(s"ROC on test data = $roc")

// COMMAND ----------

//random forest 
val ccRF = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
  .addGrid(rf2.numTrees,Array(5,10,12,15,20))
  .addGrid(rf2.maxDepth,Array(5,10,15,20))
  .build()

val evaluator = new BinaryClassificationEvaluator()
  .setMetricName("areaUnderROC")
  .setRawPredictionCol("prediction")
  .setLabelCol("label")


val cv = new CrossValidator()
  .setEstimator(rf)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

val cvModel = cv.fit(training)

val predictions = cvModel.transform(test)
predictions.select("prediction", "label", "features").show(10)

val roc = evaluator.evaluate(predictions)
println(s"ROC on test data = $roc")

// COMMAND ----------

/* 
Q: Compare the models' performances

We think the random forest is the best model for this dataset since the result of ROC on test data presents 0.7628(random forest) > 0.6544(decision tree) > 0.5(logistic regression). 
---------
Q: Defend your choice of best model

For logistic regression, its outputs are mainly based on probabilistic, and it can be regularized to avoid overfitting. However, it doesn't perform well in multiple classifications. But this problem would not exist in our dataset because we only have two classes in the label. 

For decision trees, it requires less effort in pre-processing. But decision tree often involves higher time to train the model. Also, it's sensitive to small changes in the dataset.

As to the random forest, it is able to deal with unbalanced and missing data and provide powerful predictions. Random Forest weaknesses are that when used for regression they cannot predict  beyond the range in the training data, and that they may over-fit data sets that are particularly noisy. In our model, random forest requires longest time, while decision tree needs shortest time. 

*/

// COMMAND ----------

/*
Q: What more would you do with this data? Anything to help you devise a better solution?

First we drop the ID column, and test if there is any missing value in each column, then we used PCA to do Data dimensionality reductiondo then do the starnderliation. 
For each kind of features, we computing the hyperparameter tuning by using CrossValidator. We tries different numTrees and maxDepth many times.
Then we could obtain the result with 0.7723518872617805(random forest) > 0.6363167977986756(decision tree) > 0.5(logistic regression).

Though our method didn't improve much than the stander algorithm, we decreased the features' number from 24 to 19, which would save more time when training the model. 
*/
