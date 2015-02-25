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
		- It reads patches from a text file and creates a distributed dataset (RDD). 
		- Runs some pre-processing steps on the patches in parallel: contrast normalization and ZCA whitening. 
		- Runs the parallel K-means implementation of Spark on the pre-processed patches.
		- Extracts the centers (filters) learned through K-means.
		- Max-pools the learned filters again with K-means to compress very similar filters together and reduce the final feature representation.
		- Reads new dataset with larger patches for the feature extraction step. 
		- It computes the new feature representations in parallel.
		- It compresses the feature representation with max-pooling.

	- PreProcess.java: file that contains classes for patch pre-processing: contrast normalization, ZCA whitening, mean subtraction, etc.
	- MatrixOps.java: class that contains matrix and vector manipulations.
	- Config.java: auxiliary class that contains a problem configuration to be passed as argument to a map or reduce call. It contains necessary parameters of the algorithms.
	- FeatureExtraction.java: class that implements a parallel feature extraction process using a the map call of Spark.  