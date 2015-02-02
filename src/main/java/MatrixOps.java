import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;


/* helper class for base matrix operations

Available methods:
	- transpose: returns the transpose of a matrix
	- getRow: returns a specific row from a matrix 
	- getCol: returns a specific column from a matrix
	- toString: overrides the toString() method of the classes Vector and Matrix

	- VecNormSq: squared norm of a vector
	- MatNormSq: squared norm of a matrix 
	- VecDistSq: squared euclidean distance between two vectors
	- MatDistSq: squared euclidean distance between two matrices
	- ComputeDistancesSq: Squared pair-wise distances between rows of Matrices

	- DiagMatMatMult: Multiplication of a matrix by a diagonal matrix from the left
	- MatMatMult: Matrix-Matrix multiplication
	- MatVecMult: Matrix-Vector multiplication
	- VecVecMultIn: Inner product between two vectors
	- VecVecMultOut: Outer vector product

*/
public class MatrixOps {

	// override the method apply()
	/*@Override public double apply(int i, int j) {
		return this.apply(int i, int j);
	}*/

	// Override the method numCols()
	/*@Override public int numCols() {
		return this.numCols();
	}

	// Override the method numRows()
	@Override public int numRows() {
		return this.numRows();
	}

	// Override the method toArray()
	@Override public double[] toArray() {
		return this.toArray();
	}

	// Override the method toBreeze() 
	@Override public breeze.linalg.Matrix<Object> toBreeze() {
		return this.toBreeze();
	}*/

	// vectorize a matrix
	// TODO::

	// method that returns the transpose of a matrix
	public Matrix transpose(Matrix M) {
		
		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the transpose
		double[] Mt = new double[n*m];

		// perform the transposition
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				Mt[j+m*i] = M.apply(i, j);
			}
		}

		return Matrices.dense(m, n, Mt);
	}

	
	// method that reshapes the input vector to a matrix with specified dimensions
	public Matrix reshapeVec2Mat(Vector v, int[] dims) {

		return Matrices.dense(dims[0], dims[1], v.toArray()); 
	}


	// method that reshapes the input matrix to a matrix
	public Matrix reshapeMat2Mat(Matrix M, int[] dims) {

		return Matrices.dense(dims[0], dims[1], M.toArray());
	}


	// method that converts an array of vectors to a matrix
	public Matrix convertVectors2Mat(Vector[] V, int k) {

		// size of the vectors inside the array
		int s = V[0].size();

		// allocate memory for output matrix
		double[] out = new double[s*k];
		
		// assign vectors to each row of the matrix
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < s; j++) {
				out[i+k*j] = V[i].apply(j);
			}
		}

		return Matrices.dense(k, s, out);
	}


	// method that vectorizes column-major the input matrix
	public Vector vec(Matrix M) {

		// size of the matrix
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the vector
		double[] out = new double[m*n];

		// global counter of the vector
		int c = 0;

		// main assignment loop 
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				out[c] = M.apply(i, j);
				c++;	
			}
		}

		return Vectors.dense(m*n, out);
	}


	// get a specific row from a matrix, index starts from zero
	public Vector getRow(Matrix M, int r) throws IndexOutOfBoundsException {
		
		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// check for the index bounds
		if (r >= n) {
			throw new IndexOutOfBoundsException("Row index argument is out of bounds!");
		}

		// return the specified row
		double[] row = new double[m];
		for (int j = 0; j < m; j++) {
			row[j] = M.apply(r, j);
		}

		return Vectors.dense(row);
	}

	// get several rows from a matrix

	// get a specific column from a matrix
	public Vector getCol(Matrix M, int c) throws IndexOutOfBoundsException {
		
		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// check for the index bounds
		if (c >= m) {
			throw new IndexOutOfBoundsException("Column index argument is out of bounds!");
		}

		// return the specified column
		double[] col = new double[n];
		for (int i = 0; i < n; i++) {
			col[i] = M.apply(i, c);
		}
		
		return Vectors.dense(col);
	}

	// get several columns from a matrix

	
	// get mean column vector of a matrix
	public Vector meanColVec(Matrix M) {

		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the mean vector and temporary sum vector
		double[] out = new double[n];

		// compute mean vector
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				sum += M.apply(i ,j);
			}
			out[i] = sum / m;
			sum = 0.0;
		}

		return Vectors.dense(out);
	}


	// get mean row vector of a matrix
	public Vector meanRowVec(Matrix M) {

		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the mean vector and temporary sum vector
		double[] out = new double[m];

		// compute mean vector
		double sum = 0.0;
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				sum += M.apply(i ,j);
			}
			out[j] = sum / n;
			sum = 0.0;
		}

		return Vectors.dense(out);
	}


	// override toString() function for printing an Object of the Vector class
	public String toString(Vector v) {
		
		// vector length
		int s = v.size();

		// StringBuilder allocates memory from before, better performance than 
		// appending the string every time
		StringBuilder out = new StringBuilder(s*32);
		for (int i = 0; i < s; i++) {
			out.append(v.apply(i));
			out.append("\t");	// tab between any element in the row
		}

		return out.toString();
	}


	// override toString() function for printing an Object of the Matrix class
	public String toString(Matrix M) {
		
		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// StringBuilder allocates memory from before, better performance than 
		// appending the string every time
		StringBuilder out = new StringBuilder(n*m*32);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				out.append(M.apply(i, j));
				out.append("\t");	// tab between any element in the row
			}
			out.append("\n");	// newline for the next row
		}

		return out.toString();
	}


	// squared l2 norm of a vector
	public double VecNormSq(Vector v) {
	
		// vector length
		int s = v.size();

		// sum of squares
		double normSq = 0.0;
		for (int i = 0; i < s; i++) {
			normSq += v.apply(i) * v.apply(i);
		}

		return normSq;
	}


	// squared l2 norm of a matrix (sum of squared values) 
	public double MatNormSq(Matrix M) {
	
		// size of the matrix
		int n = M.numRows();
		int m = M.numCols();
		
		// sum of squares
		double normSq = 0.0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				normSq += M.apply(i, j) * M.apply(i, j);
			}
		}

		return normSq;
	}


	// compute squared l2 distance between two vectors
	public double VecDistSq(Vector v1, Vector v2) throws IllegalArgumentException {
	
		// length of the first vector
		int s1 = v1.size();

		// length of the second vector, it should have the same size!
		int s2 = v2.size();

		if (s1 != s2) {
			throw new IllegalArgumentException("The two vectors do not have the same length!");
		}

		// sum of element-wise squared differences
		double distSq = 0.0;
		for (int i = 0; i < s1; i++) {
			distSq += (v1.apply(i) - v2.apply(i)) * (v1.apply(i) - v2.apply(i));
		}

		return distSq;
	}


	// compute squared l2 distance between two matrices
	public double MatDistSq(Matrix A, Matrix B) throws IllegalArgumentException {
	
		// dimensions of the first matrix
		int n1 = A.numRows();
		int m1 = A.numCols();

		// dimension of the second matrix, should be the same as the first!
		int n2 = B.numRows();
		int m2 = B.numCols();

		if ((n1 != n2) || (m1 != m2)) {
			throw new IllegalArgumentException("The two matrices do not have the same length!");
		}

		// sum of element-wise squared differences
		double distSq = 0.0;
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < m1; j++) {
				distSq += (A.apply(i, j) - B.apply(i, j)) * (A.apply(i, j) - B.apply(i, j));
			}
		}

		return distSq;
	}


	// compute sqaured distances between row vectors in two different matrices
	// the return argument will be a matrix with pair-wise distances 
	public Matrix ComputeDistancesSq(Matrix A, Matrix B) throws IllegalArgumentException {
		
		// dimensions of the first matrix
		int n1 = A.numRows();
		int m1 = A.numCols();

		// dimensions of the second matrix, the number of columns should be the same
		int n2 = B.numRows();
		int m2 = B.numCols();

		if (m1 != m2) {
			throw new IllegalArgumentException("The two matrices should have the same number of columns!");
		}

		// compute pair-wise distances
		double[] distSq = new double[n1*n2];
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < n2; j++) {
				distSq[i+n1*j] = VecDistSq(getRow(A, i), getRow(B, j));
			}
		}
		
		return Matrices.dense(n1, n2, distSq);
	}


	// multiply a diagonal matrix (vector in reality) with a matrix
	public Matrix DiagMatMatMult(Vector v, Matrix M) throws IllegalArgumentException {
	
		// size of the matrix, vector's size is the same as the matrix rows
		int n = M.numRows();
		int m = M.numCols();

		// vector length
		int s = v.size();
		
		// throw an exception if sizes are not compatible
		if (n != s) {
			throw new IllegalArgumentException("Incompatible vector and matrix sizes!");
		}

		// allocate memory for the output matrix
		double[] out = new double[n*m];
	
		// perform the multiplication
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				out[i+n*j] = v.apply(i) * M.apply(i, j);
			}
		}

		return Matrices.dense(n, m, out);
	}


	// multiply a matrix with a matrix from the left
	public Matrix MatMatMult(Matrix A, Matrix B) throws IllegalArgumentException  {
		
		// size of the first matrix
		int n = A.numRows();
		int m = A.numCols();

		// columns of the second matrix
		int p = B.numRows();
		int r = B.numCols();

		// throw an exception if matrix sizes are not compatible
		if (m != p) {
			throw new IllegalArgumentException("Matrix sizes are incompatible!"); 
		}
	
		// allocate memory for the output matrix
		double[] out = new double[n*r];
		
		// perform the multiplication
		double s = 0.0;
		for (int i = 0; i < n; i++ ) {
			for (int j = 0; j < r; j++) {
				for (int k = 0; k < m; k++) {
					s += A.apply(i, k) * B.apply(k, j);
				}
			
				// the final inner product is the resulting (i,j) entry
				out[i+n*j] = s;
				s = 0.0;
			}
		}
		
		return Matrices.dense(n, r, out);
	}

	
	// compute all overlapping patches of a 2-D matrix
	// each patch is a column of the resulting matrix
	public Matrix im2col(Matrix M, int[] blockSize) {

		// size of the matrix 
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the final output matrix 
		// which contains in each column an overlapping patch
		int blockSizeTotal = blockSize[0] * blockSize[1];
		int[] sizeSmall = {n-blockSize[0]+1,  m-blockSize[1]+1};
		int numPatches = sizeSmall[0] * sizeSmall[1];
		
		double[][] out = new double[blockSizeTotal][numPatches];

		// main loop for patch extraction
		int countPatch = 0;
		int countDim = 0;
		for (int j = 0; j < sizeSmall[1]; j++) {
			for (int i = 0; i < sizeSmall[0]; i++) {	

				// loop over the block
				for (int l = 0; l < blockSize[1]; l++) {
					for (int k = 0; k < blockSize[0]; k++) {
						out[countDim][countPatch] = M.apply(i+k,j+l);
						countDim++;
					}
				}
				countPatch += 1;
				countDim = 0; 
			}
		}

		// vectorize the matrix and convert it to Apache format
		int totalNum = 0;
		double[] outVec = new double[blockSizeTotal*numPatches];
		for (int j = 0; j < numPatches; j++) {
			for (int i = 0; i < blockSizeTotal; i++) {
				outVec[totalNum] = out[i][j];
				totalNum++;
			}
		}

		return Matrices.dense(blockSizeTotal, numPatches, outVec);
	}


	// 2-D convolution, without any zero-padding, resulting image of smaller size than original
	public Matrix conv2(Matrix M, Matrix F) {

		// size of the matrix to be convolved
		int n = M.numRows();
		int m = M.numCols();

		// size of kernel
		int r = F.numRows();
		int c = F.numCols();

		// allocate memory for the convolution result
		int oR = n - r + 1;
		int oC = m - c + 1;
		double[] out = new double[oR*oC];	

		// 2-D convolution
		double s = 0.0;
		for (int i = 0; i < oR; i++) {
			for (int j = 0; j < oC; j++) {

				// loop over the filter, multiply and sum up
				for (int k = 0; k < r; k++) {
					for (int l = 0; l < c; l++) {
						s += F.apply(k,l) * M.apply(i+k,j+l);
					}
				}
				out[i+oR*j] = s;
				s = 0.0;	
			}
		}

		return Matrices.dense(oR, oC, out);
	}


	// compute contrast normalization on a local vector
	public Vector localVecContrastNormalization(Vector v, double e) {	

		// vector size
		int s = v.size();

		// compute mean value of the vector
		double m = 0;
		for (int i = 0; i < s; i++) {
			m += v.apply(i);
		}
		m /= s;

		// compute standard deviation of the vector
		double stdev = 0;
		for (int i = 0; i < s; i++) {
			stdev += (v.apply(i) - m) * (v.apply(i) - m);
		}
		stdev = stdev / (s - 1);

		// subtract mean and divide by the standard deviation
		//double e = 10;	// HERE change this!!!!
		for (int i = 0; i < s; i++) {
			v.toArray()[i] = v.apply(i) - m;
			v.toArray()[i] = v.apply(i) / Math.sqrt((stdev + e));
		}

		return v;
	}


	// compute contrast normalization on a local matrix, column by column
	public Matrix localMatContrastNormalization(Matrix M, double e) {

		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for temporary vector and final matrix
		Vector cur = Vectors.dense(new double[m]);		
		Matrix C = Matrices.dense(n, m, new double[n*m]);

		// main loop for contrast normalization
		for (int i = 0; i < n; i++) {
			cur = getRow(M, i);
			cur = localVecContrastNormalization(cur, e);

			// copy the normalized row back to the result
			for (int j = 0; j < m; j++) {
				C.toArray()[i+n*j] = cur.toArray()[j];
			}
		}

		return C;
	}


	// subtract mean from a local vector
	public Vector localVecSubtractMean(Vector v, Vector m) {

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


	// subtract mean from a local matrix row by row
	public Matrix localMatSubtractMean(Matrix M, Vector v) {

		// maybe here check the size of the vector and throw an exception!!
		
		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for temporary vector and final matrix
		Vector cur = Vectors.dense(n, new double[m]);	
		Matrix C = Matrices.dense(n, m, new double[n*m]);

		// loop over elements to subtract the mean row by row
		for (int i = 0; i < n; i++) {
			cur = getRow(M, i);
			cur = localVecSubtractMean(cur, v);

			// copy the subtracted row back to the result
			for (int j = 0; j < m; j++) {
				C.toArray()[i+n*j] = cur.toArray()[j];
			}
		}

		return C;
	}


	// max-pool an image
	public Matrix pool(Matrix M, int[] poolSize) {

		// matrix size
		int n = M.numRows();
		int m = M.numCols();

		// pooled matrix size
		int[] poolDim = new int[2];
		poolDim[0] = (int) Math.floor(n/poolSize[0]);
		poolDim[1] = (int) Math.floor(m/poolSize[1]);
		double[] out = new double[poolDim[0]*poolDim[1]];	

		
		// upper left patch indices
		int k = 0;
		int l = 0;
		
		// lower right patch indices		
		int kk = 0;
		int ll = 0;

		// main loop for 2-D max pooling
		for (int i = 0; i < poolDim[0]; i++) {
			for (int j = 0; j < poolDim[1]; j++) {

				System.out.println("Here OK " + i + " " + j);
				// extract the current patch lower right indices
				kk = Math.min(k+poolSize[0],n);
				ll = Math.min(l+poolSize[1],m);

				// compute max in the current patch
				double maxPatch = -Double.MAX_VALUE;
				for (int p = k; p < kk; p++) {
					for (int q = l; q < ll; q++) {
						if (M.apply(p, q) > maxPatch) {
							maxPatch = M.apply(p, q);	
						}
					}
				}

				// assign the max value in the resulting matrix
				out[i+n*j] = maxPatch;
				
				// go one step to the right
				l += poolSize[1];
			}
			
			// go one step down
			k += poolSize[0];
			l = 0;
		}

		return Matrices.dense(poolDim[0], poolDim[1], out);
	}


	// matrix vector multiplication
	//public Matrix MatVecMult(Matrix A, Vector x) throws IllegalStateException {}
	
	// inner product between two vectors
	//public Vector VecVecMultIn(Vector v1, Vector v2) throws IllegalStateException {}

	// outer product between two vectors
	//public Matrix VecVecMultOut(Vector v1, Vector v2) throws IllegalStateException {}
}

