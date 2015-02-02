import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

/* Class that is used for gathering input arguments for map and reduce functions.

Available methods:
	- 
*/

public class Config {

	// private variables that make up the input configuration
	private int[] dims;
	private Vector[] filters;
	private Matrix zca;
	private Vector m;
	private int[] poolSize;
	private int[] rfSize;
	private int k;
	private double eps1;
	private double eps2;

	// constructor that initializes the private variables
	public Config(int[] dimsIn, Vector[] filtersIn, Matrix zcaIn, Vector mIn, int[] poolSizeIn, int[] rfSizeIn, int kIn, double eps1In, double eps2In) {
		dims = dimsIn;
		filters = filtersIn;
		zca = zcaIn;
		m = mIn;
		poolSize = poolSizeIn;
		rfSize = rfSizeIn;
		k = kIn;
		eps1 = eps1In;
		eps2 = eps2In;
	}


	/* set functions */

	// set input image dimensions
	public void setDims(int[] dimsIn) {
		dims = dimsIn;
	}

	// set learned filters
	public void setFilters(Vector[] filtersIn) {
		filters = filtersIn;
	}

	// set ZCA matrix
	public void setZCA(Matrix zcaIn) {
		zca = zcaIn;
	}

	// set mean patch
	public void setMean(Vector mIn) {
		m = mIn;
	}

	// set size of the pooling block
	public void setPoolSize(int[] poolSizeIn) {
		poolSize = poolSizeIn;
	}

	// set size of the receptive field
	public void setRfSize(int[] rfSizeIn) {
		rfSize = rfSizeIn;
	}

	// set number of filters
	public void setK(int kIn) {
		k = kIn;
	}

	// set eps1 for contrast normalization
	public void setEps1(double eps1In) {
		eps1 = eps1In;
	}

	// set eps2 for ZCA whitening
	public void setEps2(double eps2In) {
		eps2 = eps2In;
	}

	/* get functions */

	// get input image dimensions
	public int[] getDims() {
		return dims;
	}

	// get learned filters
	public Vector[] getFilters() {
		return filters;	
	}

	// get ZCA matrix
	public Matrix getZCA() {
		return zca;
	}

	// get the mean of the training patches
	public Vector getMean() {
		return m;
	}

	// get size of pooling block
	public int[] getPoolSize() {
		return poolSize;
	}

	// get size of receptive field
	public int[] getRfSize() {
		return rfSize;
	}

	// get number of filters
	public int getK() {
		return k;
	}

	// get eps1 for contrast normalization
	public double getEps1() {
		return eps1;
	}

	// get eps2 for ZCA whitening
	public double getEps2() {
		return eps2;
	}
}