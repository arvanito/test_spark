import java.lang.Math;

import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;


/* Class that performs convolutional feature extraction on image data
	
Available methods:
	- 


public class FeatureExtraction {

	// function that extracts features from each data point in the dataset
	public Vector call(Vector v, Config conf) {

		// number of features learned
		int numFeatures = conf.getK();

		// filters
		Vector[] D = conf.getFilters();

		// number of filters
		int k = conf.getK();

		// dimensions of input image data
		int[] dims = conf.getDims();

		// receptive field size
		int[] rfSize = conf.getRfSize();

		// pool size
		int[] poolSize = conf.getPoolSize();

		// ZCA matrix
		Matrix zca = conf.getZCA();

		// mean from ZCA
		Vector zcaMean = conf.getMean();

		// epsilons for pre-processing
		double eps1 = conf.getEps1();
		double eps2 = conf.getEps2();

		// compute dimensions of the convolved features
		//int dim1 = (int) Math.floor((dims[0]-rfSize[0]+1)/poolSize[0]);
		//int dim2 = (int) Math.floor((dims[1]-rfSize[1]+1)/poolSize[1]);

		// create all overlapping patches
		MatrixOps matrixOps = new MatrixOps();
		Matrix MD = matrixOps.convertVectors2Mat(D, k);	// convert array of vectors to matrix
		Matrix M = matrixOps.reshapeVec2Mat(v, dims);	// reshape the vector
		Matrix patches = matrixOps.im2col(M, rfSize);			
		patches = matrixOps.transpose(patches);

		// pre-process the patches with contrast normalization and ZCA whitening
		patches = matrixOps.localMatContrastNormalization(patches, eps1);
		patches = matrixOps.localMatSubtractMean(patches, zcaMean);
		patches = matrixOps.MatMatMult(patches, zca);

		// compute activation of patches
		patches = matrixOps.MatMatMult(patches, matrixOps.transpose(MD));

		// pool 


	}
}*/