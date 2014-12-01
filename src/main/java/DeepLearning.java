// Imports
import java.io.File;
import java.io.IOException;
import java.util.*;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.ArrayUtils;

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


// main class for pre-processing of patches                      
public class DeepLearning {

	public static void main(String[] args) {
	
		
  }

}

