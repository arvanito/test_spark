import java.io.Serializable;
import java.lang.Math;

import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;


/* Class that performs convolutional feature extraction on image data
	
Available methods:
	- call: map function that extracts features on each data point
*/

public class FeatureExtraction implements Serializable {

	// function that extracts features from each data point in the dataset
	public Vector call(Vector v, Config conf) {

		// number of features learned
		int numFeatures = conf.getK();

		// filters
		DenseMatrix D = conf.getFilters();

		// number of filters
		int k = conf.getK();

		// dimensions of input image data
		int[] dims = conf.getDims();

		// receptive field size
		int[] rfSize = conf.getRfSize();

		// pool size
		int[] poolSize = conf.getPoolSize();

		// ZCA Matrix
		DenseMatrix zca = conf.getZCA();

		// mean from ZCA
		DenseVector zcaMean = conf.getMean();

		// epsilons for pre-processing
		double eps1 = conf.getEps1();
		double eps2 = conf.getEps2();

		// get number of groups
		int numGroups = conf.getNumGroups();

		// group assignments from K-means on learned filters
		Integer[] groups = conf.getGroups();

		// compute dimensions of the convolved features
		//int dim1 = (int) Math.floor((dims[0]-rfSize[0]+1)/poolSize[0]);
		//int dim2 = (int) Math.floor((dims[1]-rfSize[1]+1)/poolSize[1]);

		// create all overlapping patches
		MatrixOps matrixOps = new MatrixOps();
		//DenseMatrix MD = matrixOps.convertDenseVectors2Mat(D, k);	// convert array of DenseVectors to Matrix
		DenseMatrix M = matrixOps.reshapeVec2Mat((DenseVector) v, dims);	// reshape the Vector
		DenseMatrix patches = matrixOps.im2col(M, rfSize);	
		patches = matrixOps.transpose(patches);

		// pre-process the patches with contrast normalization and ZCA whitening
		patches = matrixOps.localMatContrastNormalization(patches, eps1);
		patches = matrixOps.localMatSubtractMean(patches, zcaMean);
		patches = patches.multiply(zca);

		// compute activation of patches
		// CHANGE THIS!!!!!
		patches = patches.multiply(matrixOps.transpose(D));

		/*** pool the activation to reduce dimensionality and create invariance ***/

		// final pooled dimensions
		int dim1 = (int) Math.round(dims[0] - rfSize[0] + 1) / poolSize[0];
		int dim2 = (int) Math.round(dims[1] - rfSize[1] + 1) / poolSize[1];
		int[] pooledDims = {dim1,dim2};

		// convolved dimensions
		int[] convDims = {dims[0]-rfSize[0]+1,dims[1]-rfSize[1]+1};

		// allocate memory for pooled features
		double[] pooledPatches = new double[dim1*dim2*k];

		// do pooling for each learned filter	
		// global counter
		int gc = 0;		
		for (int f = 0; f < k; f++) {
			
			// extract current column and reshape it to a Matrix
			DenseVector patchCol = matrixOps.getCol(patches, f);
			DenseMatrix patchM = matrixOps.reshapeVec2Mat(patchCol, convDims);	

			// pool and assign to a row of the pooled features
			patchM = matrixOps.pool(patchM, poolSize);	

			for (int d2 = 0; d2 < dim2; d2++) {
				for (int d1 = 0; d1 < dim1; d1++) {
					pooledPatches[gc] = patchM.apply(d1,d2);
					gc++;
				}
			}
		}
		//DenseVector pooledPatchesVec = DenseVectors.dense(pooledPatches);
		DenseMatrix pooledPatchesMat = new DenseMatrix(dim1*dim2, k, pooledPatches);

		// apply group pooling for the learned filters
		DenseVector pooledPatchesVec = matrixOps.groupPool(pooledPatchesMat, pooledDims, k, groups, numGroups);

		/*** End of Pooling ***/

		return pooledPatchesVec;
	}
}