// Imports
import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;

import org.apache.spark.SparkConf;


// class to compute sum of two spark Vectors
// Used in a reduce call for calculating sum vectors, mean vectors, e.t.c.
/*class VectorSum implements Function2<Vector, Vector, Vector> {

	// method to compute the sum of two Vectors of the same size
  	public Vector call(Vector v1, Vector v2) { 
		// maybe here check the size of the vector and throw an exception!!
		
		// vector size
		int s = v1.size();

		// loop over elements to add the two vectors
		double[] v = new double[s];
		for (int i = 0; i < s; i++) {
			v[i] = v1.apply(i) + v2.apply(i);
		}
		
		// create dense vector from a double array
		return Vectors.dense(v);
	}
}*/




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


// main class with a K-means example
public class DeepLearning {

	public static void main(String[] args) {
	
		// create a Spark Configuration and Context
    	SparkConf conf = new SparkConf().setAppName("Deep Learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
		
    	// Load and parse data
		String path = "/Users/nikolaos/Desktop/patches.txt";		// make this a function argument!!!
    	JavaRDD<String> data = sc.textFile(path);
    	JavaRDD<Vector> parsedData = data.map(new ParseData());
		// maybe cache it here!!
		//parsedData.cache();

		// number of data points in the dataset
		long n = parsedData.count();
		
		//RDD<Vector> scalaData = parsedData.rdd();

		// contrast normalization, use lambda expression
		Double e1 = 10.0;		// make this a function argument!!!
		JavaRDD<Vector> contrastNorm = parsedData.map(x -> new ContrastNormalization().call(x, e1));
		
		// convert the JavaRRD<Vector> to a distributed RowMatrix (through Scala RDD<Vector>)
		RowMatrix patches = new RowMatrix(contrastNorm.rdd());

		// check if this mean vector is the same as the one below
		MultivariateStatisticalSummary summary = patches.computeColumnSummaryStatistics();
		Vector m = summary.mean();

		// compute the mean vector of the whole dataset
		/*Vector m = contrastNorm.reduce(new VectorSum());
		for (int i = 0; i < m.size(); i++)
			m.toArray()[i] = m.apply(i) / n;
		*/

		// remove the mean from the dataset, use lambda expression
		JavaRDD<Vector> centralContrastNorm = contrastNorm.map(x -> new SubtractMean().call(x, m));		
		patches = new RowMatrix(centralContrastNorm.rdd());
		
		// perform ZCA whitening and project the data onto the decorrelated space
		double e2 = 0.1;		// make this a function argument!!!
		Matrix ZCA = new PreProcess().performZCA(patches, e2);
		patches = patches.multiply(ZCA);

		// create a file with the processed patches and save it 
		String patchesDirString = "/Users/nikolaos/Desktop/processedPatches";
		try {
			File patchesDir = new File(patchesDirString);

			// if the directory exists, delete it and create it again
			if (patchesDir.isDirectory()) {
				FileUtils.cleanDirectory(patchesDir);
				FileUtils.forceDelete(patchesDir);
				FileUtils.forceMkdir(patchesDir);
			}
		} catch (IOException ex) {
			System.out.println(ex.toString());
		}
			
		// here: WTF is this classTag!!??!?!?!?!?
		JavaRDD<Vector> processedPatches = new JavaRDD(patches.rows(),centralContrastNorm.classTag());
		//processedPatches.saveAsTextFile(patchesDirString);

		// convert to String and save it to a .txt file, catch IOException
		//String matString = new MatrixOps().toString(ZCA);
		/*try {
			File file = new File("/Users/nikolaos/Desktop/zca.txt");		// make this a function argument!!!
			FileUtils.writeStringToFile(file, matString);
		} catch (IOException ex) {
			System.out.println(ex.toString());
		}*/

		// collect the data
		/*List<Vector> list = contrastNorm.collect();
		centralContrastNorm.coalesce(1).saveAsTextFile("/Users/nikolaos/Desktop/centralContrastNorm");
		*/

		/*
		// Cluster the data into two classes using KMeans, convert to Scale RDD
    	int numClusters = 2;
    	int numIterations = 20;
    	KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);
	
    	// Evaluate clustering by computing Within Set Sum of Squared Errors
    	double WSSSE = clusters.computeCost(parsedData.rdd());
    	System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
		*/
  }

}

