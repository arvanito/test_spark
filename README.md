test_spark
==========

So far, implementation of single-layer unsupervised feature learning with K-means and max-pooling. 
The implementation is done using Apache Spark. 
The learning architecture is based on the papers "Learning Feature Representations with K-means" and "Emergence of object-selective features in unsupervised feature learning". 

One single layer of the above architecture has the following form:
	- Patch Extraction -> Patch pre-processing -> K-means filter learning -> Feature Extraction -> Max-Pooling over features/filters -> Final feature representation
The output of one layer becomes the input to the next one, etc.

Basic files in the program:

	- DeepLearning.java: Entry point of the algorithm. It calls the appropriate methods for the following processing: 
		1. It reads patches from a text file and creates a distributed dataset (RDD). 
		2. Runs some pre-processing steps on the patches in parallel: contrast normalization and ZCA whitening. 
		3. Runs the parallel K-means implementation of Spark on the pre-processed patches.
		4. Extracts the centers (filters) learned through K-means.
		5. Max-pools the learned filters again with K-means to compress very similar filters together and reduce the final feature representation.
		6. Reads new dataset with larger patches for the feature extraction step. 
		7. It computes the new feature representations in parallel.
		8. It compresses the feature representation with max-pooling.

	- PreProcess.java: file that contains classes for patch pre-processing: contrast normalization, ZCA whitening, mean subtraction, etc.
	- MatrixOps.java: class that contains matrix and vector manipulations.
	- Config.java: auxiliary class that contains a problem configuration to be passed as argument to a map or reduce call. It contains necessary parameters of the algorithms.
	- FeatureExtraction.java: class that implements a parallel feature extraction process using a the map call of Spark.  

To compile the package, you run the following command:
> mvn package

To run locally, you do:
> path_to_spark/bin/spark-submit --class "DeepLearning" --master local[2] target/deep-learning-1.0.jar inputFile1 inputFile2 outputFilePatches eps1 eps2 k1 k2 iter outputFileCenters outputFileFeatures

where,
path_to_spark: complete path to your spark installation
inputFile1: path to the first input file (small patches)
inputFile2: path to the second input file (larger patches for feature extraction)
outputFilePatches: path to the pre-processed patches (it actually creates a folder and puts file parts inside, it should not exist, otherwise an exception is thrown)
eps1: first parameter for pre-processing
eps2: second parameter for pre-processing
k1: number of filters for the first k-means run
k2: number of filters for the second k-means run
iter: number of iterations for the two k-means runs (it is the same for now, it can change later)
outputFileCenters: path to file that will contain the learned filters (after the first k-means run)
outputFilePatches: resulting feature representation from the second input dataset (creates a folder and saves file parts inside, it should not exist, otherwise an exception is thrown)

An example run:
> ~/Documents/Spark/spark-1.2.0/bin/spark-submit --class "DeepLearning" --master local[2] target/deep-learning-1.0.jar /Users/nikolaos/Desktop/spark_test_data/patches.txt /Users/nikolaos/Desktop/spark_test_data/patches_64.txt /Users/nikolaos/Desktop/spark_test_data/processedPatches 10 0.1 100 50 100 /Users/nikolaos/Desktop/spark_test_data/clusterCenters.txt /Users/nikolaos/Desktop/spark_test_data/features_64