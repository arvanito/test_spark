import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

/* Class that contains methods for matrix pre-processing

Available methods:
	- performZCA: Performs ZCA Whitening in distributed RowMatrix

*/

public class PreProcess {

	// main method for ZCA whitening, input is a centralized ditributed matrix, zero mean
	public Matrix performZCA(RowMatrix mat, double e) {	
		
		// compute SVD of the data matrix
		// the right singular vectors are the eigenvectors of the covariance, do the integer casting here!!!
		SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD((int) mat.numCols(), true, 1.0E-9d);
		Matrix V = svd.V();		// right singular vectors
		Vector s = svd.s();		// singular values
		
		// the eigenvalues of the covariance are the squares of the singular values
		// add a regularizer and compute the square root
		long n = mat.numRows();	
		int ss = s.size();
		double[] l = new double[ss];
		for (int i = 0; i < ss; i++) {
			l[i] = (s.apply(i) * s.apply(i)) / (n - 1);
			l[i] = 1.0 / Math.sqrt(l[i] + e);
		}

		// create the ZCA matrix
		MatMult matMult = new MatMult();

		// first left multiplication with the transpose
		Matrix leftMult = matMult.DiagMatMatMult(Vectors.dense(l), new MatrixOps().transpose(V));
		
		// second left multiplication
		Matrix ZCA = matMult.MatMatMult(V, leftMult);

		// return the ZCA matrix
		return ZCA;
	}

	// main method for patch pre-processing, performs contrast normalization and zca whitening on the input data
	// HERE, check the return type!!!!	
	public void DataPreprocess(String[] args) {
	
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
		Double eps1 = 10.0;		// make this a function argument!!!
		JavaRDD<Vector> contrastNorm = parsedData.map(x -> new ContrastNormalization().call(x, eps1));
		
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
		double eps2 = 0.1;		// make this a function argument!!!
		Matrix ZCA = new PreProcess().performZCA(patches, eps2);
		patches = patches.multiply(ZCA);

		// create a file with the processed patches and save it 
		String patchesDirString = "/Users/nikolaos/Desktop/processedPatches";
		try {
			File patchesDir = new File(patchesDirString);

			// if the directory exists, delete it
			if (patchesDir.isDirectory()) {
				FileUtils.cleanDirectory(patchesDir);
				FileUtils.forceDelete(patchesDir);
			}
		} catch (IOException ex) {
			System.out.println(ex.toString());
		}
			
		// convert the distributed RowMatrix into a JavaRDD<Vector> 
		JavaRDD<Vector> processedPatches = new JavaRDD(patches.rows(),centralContrastNorm.classTag());
		processedPatches.saveAsTextFile(patchesDirString);

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
    	int numClusters = 6400;
    	int numIterations = 500;
    	KMeansModel clusters = KMeans.train(processedPatches.rdd(), numClusters, numIterations);
		*/

		/*
    	// Evaluate clustering by computing Within Set Sum of Squared Errors
    	double WSSSE = clusters.computeCost(parsedData.rdd());
    	System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
		*/
  }


}


// class to compute contrast normalization 
// Necessary pre-processing step for K-means to work well
// Anonymous function to be called inside a map function
class ContrastNormalization implements Function2<Vector, Double, Vector> {
	
	// method to compute contrast normalization, each row is one observation
	// r: original row vector -- data point
	// m: mean row vector of the dataset
	// v: std row vector of the dataset
	// e: regularizer
	public Vector call(Vector r, Double e) {	
		// vector size
		int s = r.size();

		// compute mean value of the vector
		double m = 0;
		for (int i = 0; i < s; i++) {
			m += r.apply(i);
		}
		m /= s;

		// compute standard deviation of the vector
		double stdev = 0;
		for (int i = 0; i < s; i++) {
			stdev += (r.apply(i) - m) * (r.apply(i) - m);
		}
		stdev = stdev / (s - 1);

		// subtract mean and divide by the standard deviation
		//double e = 10;	// HERE change this!!!!
		for (int i = 0; i < s; i++) {
			r.toArray()[i] = r.apply(i) - m;
			r.toArray()[i] = r.apply(i) / Math.sqrt((stdev + e));
		}

		return r;
	}

} 


// class to remove the mean vector from the dataset
// Used in a map call to subtract the mean vector from each data point
class SubtractMean implements Function2<Vector, Vector, Vector> {

	// method to subtract the mean vector from each data point
	public Vector call(Vector v, Vector m) {
		// maybe here check the size of the vector and throw an exception!!
		
		// vector size
		int s = v.size();

		// loop over elements to subtract the two vectors
		double[] sub = new double[s];
		for (int i = 0; i < s; i++) {
			sub[i] = v.apply(i) - m.apply(i);
		}
		
		// create dense vector from a double array
		return Vectors.dense(sub);
	}

}


// Helper class to parse text data for K-means clustering.
// Each row of the .txt file contains one data point.
class ParseData implements Function<String, Vector> {

	// the main method call parses the current string (line)
	// of the dataset and creates a double vector out of it
	public Vector call(String s) {
	
		// the current line is split into several strings 
		// each string corresponds to a double
		String[] sarray = s.split("\t");	
		double[] values = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {
			values[i] = Double.parseDouble(sarray[i]);	// creates a double from a string
		}
		
		// mllib class Vectors, create a dense vector from a double array
		return Vectors.dense(values);
	}

}

