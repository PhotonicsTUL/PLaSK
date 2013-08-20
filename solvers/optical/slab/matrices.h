/**
 *  \file   matrix.h Definitions of matrices and matrix operations
 */
#ifndef PLASK__SOLVER_VSLAB_MATRIX_H
#define PLASK__SOLVER_VSLAB_MATRIX_H

#include <cstring>

#include <plask/plask.hpp>
#include "fortran.h"

namespace plask { namespace  solvers { namespace slab {

template <typename T> class MatrixDiagonal;

/// General simple matrix template
template <typename T>
class Matrix {
  protected:
    int r, c;

  public:
    T* data;                //< The data of the matrix
    int* gc;                //< the reference count for the garbage collector


    Matrix() : gc(NULL) {};

    Matrix(int m, int n) : r(m), c(n) {
        data = new T[m*n]; gc = new int;
        //logger(LOG_DEBUG_MATRIX) << "      allocating matrix " << r << "x" << c << " (" << r*c*sizeof(T) << " bytes) at " << data << "\n";
        memset(data, 0, r*c*sizeof(T));
        *gc = 1;
    }

    Matrix(const Matrix<T>& M) : r(M.r), c(M.c), data(M.data), gc(M.gc) {
        if (gc) (*gc)++;
    }

    Matrix<T>& operator=(const Matrix<T>& M) {
        if (gc) {
            (*gc)--;
            if (*gc == 0) {
                delete gc; delete[] data;
                //logger(LOG_DEBUG_MATRIX) << "      freeing " << r << "x" << c << " matrix (" << r*c*sizeof(T) << " bytes) at " << data << "\n";

            }
        }
        r = M.r; c = M.c; data = M.data; gc = M.gc; if (gc) (*gc)++;
        return *this;
    }

    Matrix(const MatrixDiagonal<T>& M) {
        r = c = M.size();
        data = new T[r*c]; gc = new int;
        //logger(LOG_DEBUG_MATRIX) << "      allocating matrix " << r << "x" << c << " (" << r*c*sizeof(T) << " bytes) at " << data << " (from diagonal)\n";
        int size = r*c;
        for (int i = 0; i < size; i++) data[i] = 0;
        for (int j = 0, n = 0; j < r; j++, n += c+1) data[n] = M.data[j];
        *gc = 1;
    }

    Matrix(int m, int n, T* existing_data) : r(m), c(n), gc(NULL) {
        // Create matrix using exiting data. This data is never garbage-collected
        data = existing_data;
    }

    ~Matrix() {
        if (gc) {
            (*gc)--;
            if (*gc == 0) {
                delete gc; delete[] data;
                //logger(LOG_DEBUG_MATRIX) << "      freeing " << r << "x" << c << " matrix (" << r*c*sizeof(T) << " bytes) at " << data << "\n";

            }
        }
    }

    inline const T& operator()(int m, int n) const { return data[n*r + m]; }
    inline T& operator()(int m, int n) { return data[n*r + m]; }

    inline int rows() const { return r; }
    inline int cols() const { return c; }

    Matrix<T> copy() const {
        Matrix<T> copy_(r, c);
        memcpy(copy_.data, data, r*c*sizeof(T));
        return copy_;
    }

    Matrix<T>& operator*=(T a) {
        int size = r*c; for (int i = 0; i < size; i++) data[i] *= a;
        return *this;
    }
    Matrix<T>& operator/=(T a) {
        int size = r*c; for (int i = 0; i < size; i++) data[i] /= a;
        return *this;
    }

};

//------------------------------------------------------------------------------
/// General simple diagonal matrix template
template <typename T>
class MatrixDiagonal {
  protected:
    int siz;

  public:
    int* gc;                //< the reference count for the garbage collector
    T* data;                //< The data of the matrix

    MatrixDiagonal() : gc(NULL) {}

    MatrixDiagonal(int n) : siz(n) {
        data = new T[n]; gc = new int;
        //logger(LOG_DEBUG_MATRIX) << "      allocating diagonal matrix " << siz << "x" << siz << " (" << siz*sizeof(T) << " bytes) at " << data << "\n";
        memset(data, 0, siz*sizeof(T));
        *gc = 1;
    }

    MatrixDiagonal(const MatrixDiagonal<T>& M) : siz(M.siz), data(M.data), gc(M.gc) {
        if (gc) (*gc)++;
    }

    MatrixDiagonal<T>& operator=(const MatrixDiagonal<T>& M) {
        if (gc) {
            (*gc)--;
            if (*gc == 0) {
                delete gc; delete[] data;
                //logger(LOG_DEBUG_MATRIX) << "      freeing " << siz << "x" << siz << " diagonal matrix ("<< siz*sizeof(T) << " bytes) at " << data << "\n";
            }
        }
        siz = M.siz; data = M.data; gc = M.gc; if (gc) (*gc)++;
        return *this;
    }

    ~MatrixDiagonal() {
        if (gc) {
            (*gc)--;
            if (*gc == 0) {
                delete gc; delete[] data;
                //logger(LOG_DEBUG_MATRIX) << "      freeing " << siz << "x" << siz << " diagonal matrix ("<< siz*sizeof(T) << " bytes) at " << data << "\n";
            }
        }
    }

    inline const T& operator[](int n) const { return data[n]; }
    inline T& operator[](int n) { return data[n]; }

    inline const T& operator()(int m, int n) const { return (m == n)? data[n] : 0; }
    inline T& operator()(int m, int n) { if (m !=n) throw "MatrixDiagonal::operator(): wrong index"; else return data[n]; }

    inline int size() const { return siz; }

    MatrixDiagonal<T> copy() const {
        MatrixDiagonal<T> C(siz);
        memcpy(C.data, data, siz*sizeof(T));
        return C;
    }

    MatrixDiagonal<T>& operator*=(T a) { for (int i = 0; i < siz; i++) data[i] *= a; return *this; }
    MatrixDiagonal<T>& operator/=(T a) { for (int i = 0; i < siz; i++) data[i] /= a; return *this; }

};

//**************************************************************************
// Rectangular matrix of real and complex numbers
typedef Matrix<double> dmatrix;
typedef Matrix<dcomplex> cmatrix;

// Column vector and diagonal matrix
typedef DataVector<dcomplex> cvector;
typedef MatrixDiagonal<dcomplex> cdiagonal;

//**************************************************************************
/// Multiplication operator of the matrices (using BLAS level3)
inline cmatrix operator*(const cmatrix& A, const cmatrix& B) {
    if (A.cols() != B.rows()) throw "operator*<cmatrix,cmatrix>: cannot multiply: A.cols != B.rows";
    cmatrix C(A.rows(), B.cols());
    F(zgemm)('n', 'n', A.rows(), B.cols(), A.cols(), 1., A.data, A.rows(), B.data, B.rows(), 0., C.data, C.rows());
    return C;
}

/// Multiplication operator of the matrix-vector product (using BLAS level3)
inline cvector operator*(const cmatrix& A, const cvector& v) {
    int n = A.cols(), m = A.rows();
    if (n != v.size()) throw "mult_matrix_by_vector: A.cols != v.size";
    cvector dst(m);
    F(zgemv)('n', m, n, 1., A.data, m, v.data(), 1, 0., dst.data(), 1);
    return dst;
}

/// Multiplication by the diagonal matrix (right)
template <typename T>
inline cmatrix operator*(const Matrix<T>& A, const MatrixDiagonal<T>& B) {
    if (A.cols() != B.size()) throw "operator*<cmatrix,cdiagonal>: cannot multiply: A.cols != B.size";
    cmatrix C(A.rows(), B.size());
    register int n = 0;
    int r = A.rows(), c = A.cols();
    for (register int j = 0; j < c; j++)
    {
        register T b = B[j];
        for (register int i = 0; i < r; i++, n++)
            C.data[n] = A.data[n] * b;
    }
    return C;
}

/// Multiplication by the diagonal matrix (left)
template <typename T>
inline cmatrix operator*(const MatrixDiagonal<T>& A, const Matrix<T>& B) {
    if (A.size() != B.rows()) throw "operator*<cdiagonal,cmatrix>: cannot multiply: A.size != B.rows";
    cmatrix C(A.size(), B.cols());
    register int n = 0;
    int r = B.rows(), c = B.cols();
    for (register int j = 0; j < c; j++)
        for (register int i = 0; i < r; i++, n++)
            C.data[n] = A[i] * B.data[n];
    return C;
}

/// Multiplication of matrix by diagonal in-place (replacing A)
template <typename T>
inline void mult_matrix_by_diagonal(Matrix<T>& A, const MatrixDiagonal<T>& B) {
    if (A.cols() != B.size()) throw "mult_matrix_by_diagonal: cannot multiply: A.cols != B.size";
    register int n = 0;
    int r = A.rows(), c = A.cols();
    for (register int j = 0; j < c; j++) {
        register T b = B[j];
        for (register int i = 0; i < r; i++, n++)
            A.data[n] *= b;
    }
}

/// Multiplication of diagonal by matrix in-place (replacing B)
template <typename T>
inline void mult_diagonal_by_matrix(const MatrixDiagonal<T>& A, Matrix<T>& B) {
    if (A.size() != B.rows()) throw "mult_diagonal_by_matrix: cannot multiply: A.size != B.rows";
    register int n = 0;
    int r = B.rows(), c = B.cols();
    for (register int j = 0; j < c; j++)
        for (register int i = 0; i < r; i++, n++)
            B.data[n] *= A[i];
}

//**************************************************************************
// BLAS wrappers for multiplications without allocating additional storage
inline void mult_matrix_by_vector(const cmatrix& A, const cvector& v, cvector& dst) {
    int m, n;
    if ((n = A.cols()) != v.size()) throw "mult_matrix_by_vector: A.cols != v.size";
    if ((m = A.rows()) != dst.size()) throw "mult_matrix_by_vector: A.rows != dst.size";
    F(zgemv)('n', m, n, 1., A.data, m, v.data(), 1, 0., dst.data(), 1);
}

inline void mult_matrix_by_matrix(const cmatrix& A, const cmatrix& B, cmatrix& dst) {
    int m, n, k;
    if ((k = A.cols()) != B.rows()) throw "mult_matrix_by_matrix: cannot multiply: A.cols != B.rows";
    if ((m = A.rows()) != dst.rows()) throw "mult_matrix_by_matrix: A.rows != dst.rows";
    if ((n = B.cols()) != dst.cols()) throw "mult_matrix_by_matrix: B.cols != dst.cols";
    F(zgemm)('n', 'n', m, n, k, 1., A.data, m, B.data, k, 0., dst.data, m);
}

inline void add_mult_matrix_by_vector(const cmatrix& A, const cvector& v, cvector& dst) {
    int m, n;
    if ((n = A.cols()) != v.size()) throw "add_mult_matrix_by_vector: A.cols != v.size";
    if ((m = A.rows()) != dst.size()) throw "add_mult_matrix_by_vector: A.rows != dst.size";
    F(zgemv)('n', m, n, 1., A.data, m, v.data(), 1, 1., dst.data(), 1);
}

inline void add_mult_matrix_by_matrix(const cmatrix& A, const cmatrix& B, cmatrix& dst) {
    int m, n, k;
    if ((k = A.cols()) != B.rows()) throw "add_mult_matrix_by_matrix: A.cols != B.rows";
    if ((m = A.rows()) != dst.rows()) throw "add_mult_matrix_by_matrix: A.rows != dst.rows";
    if ((n = B.cols()) != dst.cols()) throw "add_mult_matrix_by_matrix: B.cols != dst.cols";
    F(zgemm)('n', 'n', m, n, k, 1., A.data, m, B.data, k, 1., dst.data, m);
}


// Some LAPACK wrappers
cmatrix invmult(cmatrix& A, cmatrix& B);
cvector invmult(cmatrix& A, cvector& B);
cmatrix inv(cmatrix& A);
dcomplex det(cmatrix& A);
int eigenv(cmatrix& A, cdiagonal& vals, cmatrix* rightv=NULL, cmatrix* leftv=NULL);

}}}
#endif // PLASK__SOLVER_VSLAB_MATRIX_H
