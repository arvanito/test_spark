// Imports
import java.io.File;
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
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.SparkConf;
import org.apache.spark.storage.StorageLevel;




//public class Functions {
//	static Function<Double, Function<Vector, Vector>> calculation =
//	x -> y ->  		// vector size
//	{
//		int s = r.size();
//
//		// compute mean value of the vector
//		double m = 0;
//		for (int i = 0; i < s; i++) {
//			m += r.apply(i);
//		}
//		m /= s;
//
//		// compute standard deviation of the vector
//		double stdev = 0;
//		for (int i = 0; i < s; i++) {
//			stdev += (r.apply(i) - m) * (r.apply(i) - m);
//		}
//		stdev = stdev / (s - 1);
//
//		// subtract mean and divide by the standard deviation
//		//double e = 10;	// HERE change this!!!!
//		for (int i = 0; i < s; i++) {
//			r.toArray()[i] = r.apply(i) - m;
//			r.toArray()[i] = r.apply(i) / Math.sqrt((stdev + e));
//		}
//
//		return r;
//		}
//;}

// HERE DO FUNCTION CARRYING
// class for implementing function currying the ConstrastNormalization
//class ConstrastNormalizationCurry implements Function<Double, Function<Vector,Vector>> {

	// function currying here
//	public function<> call(Double a, ) {
//		return new ConstrastNormalization()
//	}

//}


// main class for pre-processing of patches                      
public class DeepLearning {

	public static void main(String[] args) {
	
		// create a Spark Configuration and Context
    	SparkConf conf = new SparkConf().setAppName("Deep Learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
		
		// parameters, make them more user-friendly in the running environment 
		// no fixed number of arguments in every call
		// e.g., -input String -output String -eps1 Double -eps2 Double ......
		String inputFile = args[0];								// input dataset
		String outputFilePatches = args[1];						// output file for pre-processed patches
		final Double eps1 = Double.valueOf(args[2]);			// contrast normalization parameter
		final Double eps2 = Double.valueOf(args[3]);			// ZCA regularization parameter
		int numClusters = Integer.parseInt(args[4]);			// number of clusters for K-means
		int numIterations = Integer.parseInt(args[5]);			// number of iterations for K-means
		String outputFileCenters = args[6];						// output file for cluster centers
    
		// check matrix code here
		//double[] x = {1,2,3,2,3,1}

		// Load and parse data
		System.out.println("Data parsing...");
    	JavaRDD<String> data = sc.textFile(inputFile);
    	JavaRDD<Vector> parsedData = data.map(new ParseData());
		
		// assign the parsed Data to the pre-processed data
		JavaRDD<Vector> processedData = parsedData;
		//processedData = processedData.persist(new StorageLevel().MEMORY_ONLY());

		// if directory of processed patches already exists, then just go directly to k-means
		File patchesDir = new File(outputFilePatches);

		// if the directory does not exist,do pre-processing
		if (!patchesDir.isDirectory()) {
			// patch pre-processing using contrast normalization and zca whitening
			System.out.println("Data pre-processing...");
			PreProcess preProcess = new PreProcess();
			processedData = preProcess.preprocessData(processedData, outputFilePatches, eps1, eps2);
			//processedData = processedData.persist(new StorageLevel().MEMORY_ONLY());
		}

		// run K-means on the pre-processed patches, 1st layer learning
		System.out.println("K-means learning...");
    	KMeansModel patchClusters = KMeans.train(processedData.rdd(), numClusters, numIterations);
		Vector[] D1 = patchClusters.clusterCenters();  
		
		int k = patchClusters.k();		// number of clusters
		int d = D1[0].size();			// patch dimensions
		
		// convert the clusters to strings
		MatrixOps matrixOps = new MatrixOps();
		StringBuilder centersString = new StringBuilder(k*d*32);
		for (int i = 0; i < k; i++) {
			centersString.append(matrixOps.toString(D1[i]));
			centersString.append("\n");
		}
		
		// save the strings to a .txt file, one vector per row
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
		
		// feature extraction

  }

}

