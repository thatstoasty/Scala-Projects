//Import libraries and spark session
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Create a Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//Read in csv file.
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// Print the Schema of the DataFrame and check the head
data.printSchema
data.head(1)(0)

//Renaming the yearly amount spent column as the "label", also grabbing the numerical features from the data frame. This is saved as a new dataframe named df
val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership", $"Yearly Amount Spent")


// The features need to be assembled into a single "features" column to be processed by the machine learning algorithm. The vector assembler is used for this.

// Create new VectorAseembler object and pass the features to be assembled
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership", "Yearly Amount Spent")).setOutputCol("features")

//Using the assembler to transform the dataframe into one that has two columns, the labels and features
val output = assembler.transform(df).select($"label", $"features")

// Create new LinearRegression object and fit it to the data
val lr = new LinearRegression()
val lrmodel = lr.fit(output)

// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrmodel.coefficients} Intercept: ${lrmodel.intercept}")

// Show the residuals, the RMSE, the MSE, and the R^2 Values.
val trainingSummary = lrmodel.summary
trainingSummary.residuals.show()

println(s"Root Mean Squared Error: ${trainingSummary.rootMeanSquaredError}")
println(s"Mean Squared Error: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
