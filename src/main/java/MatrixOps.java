import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;


/* helper class for base matrix operations

Available methods:
	- transpose: returns the transpose of a matrix
	- toString: overrides the toString() method of the class Matrix

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

	//public MatrixOps() {}

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
		
		// create a dense Matrix from the double array
		return Matrices.dense(m, n, Mt);
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

}

/* class that contains methods for matrix-matrix multiplications
	
Available methods:
	- DiagMatMatMult: Multiplication of a matrix by a diagonal matrix from the left
	- MatMatMult: Matrix-Matrix multiplication
	- MatVecMult: Matrix-Vector multiplication
	- VecVecMultIn: Inner product between two vectors
	- VecVecMultOut: Outer vector product
*/

class MatMult {

	// multiply a diagonal matrix (vector in reality) with a matrix
	public Matrix DiagMatMatMult(Vector v, Matrix M) throws IllegalArgumentException {
	
		// size of the matrix, vector's size is the same as the matrix rows
		int n = M.numRows();
		int m = M.numCols();

		// vector length
		int s = v.size();
		
		// throw an exception if sizes are not compatible
		if (n!=s) {
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
		
		// create a dense Matrix from the double array
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
		if (m!=p) {
			throw new IllegalArgumentException("Matrix sizes are incompatible!"); 
		}
	
		// allocate memory for the output matrix
		double[] out = new double[n*r];
		
		// perform the multiplication
		double s = 0.0;
		for (int i = 0; i < n; i++ ) {
			for (int j = 0; j < r; j++) {
				for (int k = 0; k < m; k++) {
					s = s + A.apply(i, k) * B.apply(k, j);
				}
			
				// the final inner product is the resulting (i,j) entry
				out[i+n*j] = s;
				s = 0.0;
			}
		}

		// create a dense Matrix from the double array
		return Matrices.dense(n, m, out);
	}
	
	// matrix vector multiplication
	//public Matrix MatVecMult(Matrix A, Vector x) throws IllegalStateException {}
	
	// inner product between two vectors
	//public Vector VecVecMultIn(Vector v1, Vector v2) throws IllegalStateException {}

	// outer product between two vectors
	//public Matrix VecVecMultOut(Vector v1, Vector v2) throws IllegalStateException {}
}

