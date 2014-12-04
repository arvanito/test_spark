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

	// class that returns the transpose of a matrix
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
			distSq += (v1.apply(i) - v1.apply(i)) * (v1.apply(i) - v1.apply(i));
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


	// compute distances between row vectors in two different matrices
	// the return argument will be a matrix with pair-wise distances 
	public Matrix ComputeDistances(Matrix A, Matrix B) throws IllegalArgumentException {
		
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
	
	// matrix vector multiplication
	//public Matrix MatVecMult(Matrix A, Vector x) throws IllegalStateException {}
	
	// inner product between two vectors
	//public Vector VecVecMultIn(Vector v1, Vector v2) throws IllegalStateException {}

	// outer product between two vectors
	//public Matrix VecVecMultOut(Vector v1, Vector v2) throws IllegalStateException {}
}

