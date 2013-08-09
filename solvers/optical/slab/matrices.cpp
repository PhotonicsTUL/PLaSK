#include "matrices.h"

namespace plask { namespace  solvers { namespace slab {

//**************************************************************************
/// multiply inv(A)*B the A and B matrices are overwritten
cmatrix invmult(cmatrix& A, cmatrix& B)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw "invmult: cannot invert rectangular matrix";
    int N = A.rows();
    // Check if can multiply
    if (B.rows() != N)
        throw "invmult: cannot multiply matrices because of the dimensions mismatch";
    int nrhs = B.cols();
    // Needed variables
    int* ipiv = new int[N];
    int info;
    // Perform the calculation
    F(zgesv)(N, nrhs, A.data, N, ipiv, B.data, N, info);
    delete[] ipiv;
    // Return the result
    if (info > 0) throw "invmult: matrix is singular";
    return B;
}

//**************************************************************************
/// multiply inv(A)*b the A matrix and b vectors
cvector invmult(cmatrix& A, cvector& B)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw "invmult: cannot invert rectangular matrix";
    int N = A.rows();
    // Check if can multiply
    if (B.size() != N)
        throw "invmult: cannot multiply matrix by vector because of the dimensions mismatch";
    // Needed variables
    int* ipiv = new int[N];
    int info;
    // Perform the calculation
    F(zgesv)(N, 1, A.data, N, ipiv, B.data(), N, info);
    delete[] ipiv;
    // Return the result
    if (info > 0) throw "invmult: matrix is singular";
    return B;
}

//****************************************************************************
/// Invert matrix
cmatrix inv(cmatrix& A)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw "inv: cannot invert rectangular matrix";
    int N = A.rows();

    // Simply call invmult(A, I)
    cmatrix result(N, N);
    for (int i = 0; i < N; i++) result(i, i) = 1;

    invmult(A, result);

    return result;
}

//****************************************************************************
/// Find a determinant of the matrix
dcomplex det(cmatrix& A)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw "det: cannot find the determinant of rectangular matrix";
    int N = A.rows();
    // Needed variables
    int* ipiv = new int[N];
    int info;
    // Find the LU factorization
    F(zgetrf)(N, N, A.data, N, ipiv, info);
    // Ok, now compute the determinant
    dcomplex det = 1.; int p = 1;
    for (int i = 0; i < N; i++) {
        det *= A(i,i);
        if (ipiv[i] != i+1) p = -p;
    }
    delete[] ipiv;
    // Return the result
    if (p < 0) return -det; else return det;
}


//****************************************************************************
int eigenv(cmatrix& A, cdiagonal& vals, cmatrix* rightv, cmatrix* leftv)
{
    // Check the validity of the matrices
    if (A.rows() != A.cols())
        throw "eigenv: matrix A should be square";
    int N = A.rows();
    if (vals.size() != N)
        throw "eigenv: eigenvalues should have the same number of rows as the original matrix.";
    if (rightv) if (rightv->rows() != N || rightv->cols() != N)
        throw "eigenv: matrices for right eigenvectors should be square";
    if (leftv) if (leftv->rows() != N || leftv->cols() != N)
        throw "eigenv: Matrices for left eigenvectors should be square";

    // Determine the task
    char jobvl = (leftv==NULL)? 'N' : 'V';
    char jobvr = (rightv==NULL)? 'N' : 'V';

    // Determine the storage place for eigenvectors
    dcomplex* vl = (leftv==NULL)? NULL : leftv->data;
    dcomplex* vr = (rightv==NULL)? NULL : rightv->data;

    // Create the workplace
    int lwork = 2*N+1;
    //int lwork = N*N;
    dcomplex* work = new dcomplex[lwork];
    double* rwork = new double[2*N];

    // Call the lapack subroutine
    int info;
    F(zgeev)(jobvl, jobvr, N, A.data, N, vals.data, vl, N, vr, N, work, lwork, rwork, info);

    delete[] work;
    delete[] rwork;

    return info;
}

}}} // namespace plask::solvers::slab;