// Imports
import java.io.*;
import java.io.IOException;
import java.util.*;

import org.apache.commons.io.FileUtils;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.SparseMatrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.SparkConf;
import org.apache.spark.storage.StorageLevel;


// main class for pre-processing of patches                      
public class DeepLearning {

	public static void main(String[] args) {
	
		// create a Spark Configuration and Context
    	SparkConf conf = new SparkConf().setAppName("Deep Learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
		
		// parameters, make them more user-friendly in the running environment 
		// no fixed number of arguments in every call
		// e.g., -input String -output String -eps1 Double -eps2 Double ......
		String inputFile = args[0];								// input dataset 1
		String inputFile2 = args[1];							// input dataset 2
		String outputFilePatches = args[2];						// output file for pre-processed patches
		final Double eps1 = Double.valueOf(args[3]);			// contrast normalization parameter
		final Double eps2 = Double.valueOf(args[4]);			// ZCA regularization parameter
		int numClusters = Integer.parseInt(args[5]);			// number of clusters for K-means, first round 
		int numGroups = Integer.parseInt(args[6]);				// number of clusters for K-means, second round
		int numIterations = Integer.parseInt(args[7]);			// number of iterations for K-means, for both rounds, to change!!
		String outputFileCenters = args[8];						// output file for cluster centers
		String outputFileFeatures = args[9];					// output file for first layer features

		// TODO:: put these in the argument list!!!
		int[] dims = {64, 64};
		int[] poolSize = {2, 2};
		int[] rfSize = {32, 32};

		// Load and parse data
		System.out.println("Data parsing...");
    	JavaRDD<String> data = sc.textFile(inputFile);
    	JavaRDD<Vector> parsedData = data.map(new ParseData());
		
		// assign the parsed Data to the pre-processed data
		JavaRDD<Vector> processedData = parsedData;

		// matrix operation main object
		MatrixOps matrixOps = new MatrixOps();

		// create the configuration object and assign already some known values
		final Config confP = new Config();
		confP.setEps1(eps1);
		confP.setEps2(eps2);

		// if directory of processed patches already exists, then just go directly to k-means
		//File patchesDir = new File(outputFilePatches);

		// if the directory does not exist,do pre-processing
		//if (!patchesDir.isDirectory()) {
			// patch pre-processing using contrast normalization and zca whitening
		System.out.println("Data pre-processing...");
		PreProcess preProcess = new PreProcess();
		processedData = preProcess.preprocessData(processedData, outputFilePatches, confP);
		processedData = processedData.cache();
		//}

		// run K-means on the pre-processed patches, 1st layer learning
		System.out.println("K-means learning...");
    	KMeansModel patchClusters = KMeans.train(processedData.rdd(), numClusters, numIterations);
		
    	// save cluster centers in a .txt file
		Vector[] D1 = patchClusters.clusterCenters();  

		int k = patchClusters.k();		// number of clusters
		int d = D1[0].size();			// patch dimensions
		
		// convert array of Vectors to a DenseMatrix
		DenseMatrix MD = matrixOps.convertVectors2Mat(D1, k);
		confP.setFilters(MD);
		confP.setK(k);

		// convert the clusters to strings
		StringBuilder centersString = new StringBuilder(k*d*32);
		for (int i = 0; i < k; i++) {
			centersString.append(matrixOps.toString((DenseVector) D1[i]));
			centersString.append("\n");
		}

		// save the strings to a hdfs file using Hadoop
		// TODO!!!!!!!!!!!
		System.out.println("Save cluster centers to file...");
		try {
			File file = new File(outputFileCenters);		
			FileUtils.writeStringToFile(file, centersString.toString());
		} catch (IOException ex) {
			System.out.println(ex.toString());
		}

    	// Evaluate clustering by computing Within Set Sum of Squared Errors
    	double WSSSE = patchClusters.computeCost(processedData.rdd());
    	System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

		// run again k-means clustering to cluster similar learned filters together
		// use parallelize here to convert the local matrix to an RDD<Vector> for K-means !!!
		JavaRDD<Vector> clusters = sc.parallelize(Arrays.asList(D1));
		KMeansModel clusterGroups = KMeans.train(clusters.rdd(), numGroups, numIterations);

		// compute the cluster indices of the learned filters
		JavaRDD<Integer> groupsRDD = clusterGroups.predict(clusters);
		Integer[] groups = groupsRDD.collect().toArray(new Integer[k]);
		for (int i = 0; i < groups.length; i++) {
			System.out.println(groups[i]);
		}
		confP.setGroups(groups);

		// set remaining variables for feature extraction
		confP.setNumGroups(numGroups);
		confP.setDims(dims);
		confP.setPoolSize(poolSize);
		confP.setRfSize(rfSize);

  		// load patches from file, in a parallel format
     	JavaRDD<String> dataP = sc.textFile(inputFile2);
  		JavaRDD<Vector> dataPat = dataP.map(new ParseData());

     	System.out.println("Feature Extraction.....");

  		// extract features in parallel, use lambda expression
		// JavaRDD<DenseVector> dataPatFeats = dataPat.map(x -> new FeatureExtraction().call(x, confP));			
		// workaround for Java 7 compatibility!
		JavaRDD<Vector> dataPatFeats = dataPat.map(
		 	new Function<Vector, Vector>() {
		 		public Vector call(Vector x) {
		 			return new FeatureExtraction().call(x, confP);
		 		}
		 	}	
		);	
		dataPatFeats.saveAsTextFile(outputFileFeatures);
    	System.exit(0);
  }

}

