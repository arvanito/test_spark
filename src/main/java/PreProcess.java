import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

import org.apache.commons.io.FileUtils;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;

/* Class that contains methods for matrix pre-processing

Available methods:
	- performZCA: Performs ZCA Whitening in distributed RowMatrix
	- preprocessData: main function that performs data pre-processing with contrast normalization and ZCA whitening
*/
public class PreProcess implements Serializable {

	// main method for ZCA whitening, input is a centralized ditributed matrix, zero mean
	public Matrix performZCA(RowMatrix mat, final double e) {	
		
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
		MatrixOps matrixOps = new MatrixOps();

		// first left multiplication with the transpose
		Matrix leftMult = matrixOps.DiagMatMatMult(Vectors.dense(l), matrixOps.transpose(V));
		
		// second left multiplication
		Matrix ZCA = matrixOps.MatMatMult(V, leftMult);

		return ZCA;
	}

	/*
	class MyFunction<A,B> extends Function<A,B> {
		protected Object foo
		@override

		Vector call(Ve) {
			foo.run()
		}
	}*/

	// main method for patch pre-processing, performs contrast normalization and zca whitening on the input data
	// HERE, check the return type!!!!	
	public JavaRDD<Vector> preprocessData(JavaRDD<Vector> parsedData, String outputFile, final Double eps1, final Double eps2) {

		// number of data points in the dataset
		long n = parsedData.count();
		
		//RDD<Vector> scalaData = parsedData.rdd();

		// contrast normalization, use lambda expression
		//JavaRDD<Vector> contrastNorm = parsedData.map(x -> new ContrastNormalization().call(x, eps1));

		// workaround for Java 7 compatibility!
		JavaRDD<Vector> contrastNorm = parsedData.map(
			new Function<Vector, Vector>() {
  				public Vector call(Vector x) { 
  					return new ContrastNormalization().call(x, eps1);
 				}
			}
		);

		// convert the JavaRRD<Vector> to a distributed RowMatrix (through Scala RDD<Vector>)
		RowMatrix patches = new RowMatrix(contrastNorm.rdd());

		// compute mean data vector
		MultivariateStatisticalSummary summary = patches.computeColumnSummaryStatistics();
		final Vector m = summary.mean();
		
		// compute the mean vector of the whole dataset
		/*Vector m = contrastNorm.reduce(new VectorSum());
		for (int i = 0; i < m.size(); i++) {
			m.toArray()[i] = m.apply(i) / n;
		}
		String ms = new MatrixOps().toString(m);
		try {
			File msDir = new File("/Users/nikolaos/Desktop/mean.txt");
			FileUtils.writeStringToFile(msDir, ms);
			System.out.println(m);
		} catch (IOException ex) {
			System.out.println(ex.toString());
		}*/

		// remove the mean from the dataset, use lambda expression
		//JavaRDD<Vector> centralContrastNorm = contrastNorm.map(x -> new SubtractMean().call(x, m));	

		// workaround for Java 7 compatibility!
		JavaRDD<Vector> centralContrastNorm = contrastNorm.map(
			new Function<Vector, Vector>() {
				public Vector call(Vector x) {
					return new SubtractMean().call(x, m);
				}
			}	
		);	

		// create distributed matrix from centralized data, input to ZCA
		patches = new RowMatrix(centralContrastNorm.rdd());
		
		// perform ZCA whitening and project the data onto the decorrelated space
		Matrix ZCA = performZCA(patches, eps2);
		patches = patches.multiply(ZCA);

		// create a file with the processed patches and save it 
		try {
			File patchesDir = new File(outputFile);

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
		processedPatches.saveAsTextFile(outputFile);

		return processedPatches;
  }

}


// class to compute contrast normalization 
// Necessary pre-processing step for K-means to work well
// Anonymous function to be called inside a map function
class ContrastNormalization {//implements Function2<Vector, Double, Vector> {
	
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


// class to compute sum of two spark Vectors
// Used in a reduce call for calculating sum vectors, mean vectors, e.t.c.
class VectorSum {//implements Function2<Vector, Vector, Vector> {

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

		return Vectors.dense(v);
	}

}


// class to remove the mean vector from the dataset
// Used in a map call to subtract the mean vector from each data point
class SubtractMean {//implements Function2<Vector, Vector, Vector> {

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
		
		return Vectors.dense(values);
	}

}

