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

