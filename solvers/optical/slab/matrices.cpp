#include "matrices.h"

namespace plask { namespace optical { namespace slab {

cmatrix invmult(cmatrix& A, cmatrix& B)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw ComputationError("invmult", "Cannot invert rectangular matrix");
    const std::size_t N = A.rows();
    // Check if can multiply
    if (B.rows() != N)
        throw ComputationError("invmult", "Cannot multiply matrices because of the dimensions mismatch");
    const std::size_t nrhs = B.cols();
    // Needed variables
    std::unique_ptr<int[]> ipiv(new int[N]);
    int info;
    // Perform the calculation
    zgesv(int(N), int(nrhs), A.data(), int(N), ipiv.get(), B.data(), int(N), info);
    // Return the result
    if (info > 0) throw ComputationError("invmult", "Matrix is singular");
    return B;
}


cvector invmult(cmatrix& A, cvector& B)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw ComputationError("invmult", "Cannot invert rectangular matrix");
    const std::size_t N = A.rows();
    // Check if can multiply
    if (B.size() != N)
        throw ComputationError("invmult", "Cannot multiply matrix by vector because of the dimensions mismatch");
    // Needed variables
    std::unique_ptr<int[]> ipiv(new int[N]);
    int info;
    // Perform the calculation
    zgesv(int(N), 1, A.data(), int(N), ipiv.get(), B.data(), int(N), info);
    // Return the result
    if (info > 0) throw ComputationError("invmult", "Matrix is singular");
    return B;
}


cmatrix inv(cmatrix& A)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw ComputationError("inv", "Cannot invert rectangular matrix");
    const std::size_t N = A.rows();

    // Simply call invmult(A, I)
    cmatrix result(N, N, 0.);
    for (std::size_t i = 0; i < N; i++) result(i, i) = 1;

    invmult(A, result);

    return result;
}


dcomplex det(cmatrix& A)
{
    // Check if the A is a square matrix
    if (A.rows() != A.cols())
        throw ComputationError("det", "Cannot find the determinant of rectangular matrix");
    const std::size_t N = A.rows();
    // Needed variables
    std::unique_ptr<int[]> ipiv(new int[N]);
    int info;
    // Find the LU factorization
    zgetrf(int(N), int(N), A.data(), int(N), ipiv.get(), info);
    // Ok, now compute the determinant
    dcomplex det = 1.; int p = 1;
    for (std::size_t i = 0; i < N; i++) {
        det *= A(i,i);
        if (std::size_t(ipiv[i]) != i+1) p = -p;
    }
    // Return the result
    if (p < 0) return -det; else return det;
}



int eigenv(cmatrix& A, cdiagonal& vals, cmatrix* rightv, cmatrix* leftv)
{
    // Check the validity of the matrices
    if (A.rows() != A.cols())
        throw ComputationError("eigenv", "Matrix A should be square");
    const std::size_t N = A.rows();
    if (vals.size() != N)
        throw ComputationError("eigenv", "Eigenvalues should have the same number of rows as the original matrix.");
    if (rightv) if (rightv->rows() != N || rightv->cols() != N)
        throw ComputationError("eigenv", "Matrices for right eigenvectors should be square");
    if (leftv) if (leftv->rows() != N || leftv->cols() != N)
        throw ComputationError("eigenv", "Matrices for left eigenvectors should be square");

    // Determine the task
    char jobvl = (leftv==NULL)? 'N' : 'V';
    char jobvr = (rightv==NULL)? 'N' : 'V';

    // Determine the storage place for eigenvectors
    dcomplex* vl = (leftv==NULL)? NULL : leftv->data();
    dcomplex* vr = (rightv==NULL)? NULL : rightv->data();

    // Create the workplace
    const std::size_t lwork = 2*N+1;
    //int lwork = N*N;
    std::unique_ptr<dcomplex[]> work(new dcomplex[lwork]);
    std::unique_ptr<double[]> rwork(new double[2*N]);

    // Call the lapack subroutine
    int info;
    zgeev(jobvl, jobvr, int(N), A.data(), int(N), vals.data(), vl, int(N), vr, int(N), work.get(), int(lwork), rwork.get(), info);

    return info;
}

}}} // namespace plask::optical::slab;
