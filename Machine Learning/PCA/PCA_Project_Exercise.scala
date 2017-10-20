// Imports
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, PCA}
import org.apache.spark.ml.linalg.Vectors

// Create SparkSession
val spark = SparkSession.builder.appName("PCA_Example").getOrCreate()

// Use Spark to read in the Cancer_Data file.
val df = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("Cancer_Data")

df.printSchema

// simplify passing in feature columns

val colnames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
"mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
"radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
"concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
"worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
"worst concave points", "worst symmetry", "worst fractal dimension"))

//create an assembler object that will take the in the columns with the names in colnames and set the output as features. Then use the assembler to transform the dataframe
val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")
val newdata = assembler.transform(df).select($"features")

// Use StandardScaler on the data
// Create a new StandardScaler() object called scaler
// Set the input to the features column and the ouput to a column called
// scaledFeatures

//use standard scaler to standardize the data and prevent certain features from too strongly influencing the algorithm

val scaler = (new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false))

//fit the Scalermodel to the dataframe
val scalerModel = scaler.fit(newdata)

// Normalize each feature to have unit standard deviation using the transform method
val scaled_features = scalerModel.transform(newdata)

// create a new pca object that will take in the scaled features, reduce them to the principal component features and  output them.
val pca = (new PCA()
                    .setInputCol("features")
                    .setOutputCol("pca_features")
                    .setK(4)
                    .fit(newdata))

//transform the dataframe using the pca object we just created and select the pca features
val pcaDF = pca.transform(scaled_features)
val result = pcaDF.select("pca_features")
result.show()

//confirm that we only have 4 principal component features (matches our set K values of 4)
pcaDF.head(1)
