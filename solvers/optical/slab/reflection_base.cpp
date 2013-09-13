#include "reflection_base.h"
#include "fortran.h"

namespace plask { namespace solvers { namespace slab {

template <typename GeometryT>
void ReflectionSolver<GeometryT>::init()
{
    // Default values of some parameters

    if (this->interface >= this->stack.size() || this->interface == 0)
        throw BadInput(this->getId(), "Wrong interface position");

    // Reserve space for matrix multiplications...
    int N0 = diagonalizer->source().matrixSize();
    int N = diagonalizer->matrixSize();
    M = cmatrix(N0, N0);

    // ...and eigenvalues determination
    delete[] evals;
    delete[] work;
    delete[] rwork;
    evals = new dcomplex[N0];
    lwork = N0*N0;
    rwork = new double[2*N0];
    work = new dcomplex[lwork];

    // Nothing found so far
    fields_determined = false;
    interface_field = nullptr;

    this->writelog(LOG_DETAIL, "Initializing reflection-matrix solver"
                               " (%1% layers in the stack, interface after %2% layers", this->stack.size(), this->interface);

    P = cmatrix(N,N);
    phas = cdiagonal(N);
    A = cmatrix(N,N);       // A is also temporary matrix
    delete[] ipiv;
    ipiv = new int[N];
    allP = false;

}

template <typename GeometryT>
ReflectionSolver<GeometryT>::~ReflectionSolver()
{
    delete[] evals;
    delete[] work;
    delete[] rwork;
    delete[] ipiv;
}


// //******************************************************************************
// /// Return the eigenvalue of the matrix M with the smallest magnitude
// dcomplex SlabSolverBase::char_val(const dcomplex k0, const dcomplex kx, const dcomplex ky, const double mgain)
// {
//     if (target == LOSS) {
//         if (current_loglevel | LOG_SHOWCHARVAL) {
//             if (variable == FREQUENCY) {
//                 std::string msg = "    k0 = " + str(k0);
//                 for (int l = msg.length(); l < 44; l++) msg += " ";
//                 logger(LOG_SHOWCHARVAL) << msg;
//             } else if (variable == WAVEVECTOR_X) {
//                 std::string msg = "    kx = " + str(kx);
//                 for (int l = msg.length(); l < 44; l++) msg += " ";
//                 logger(LOG_SHOWCHARVAL) << msg;
//             } else { // variable == WAVEVECTOR_Y
//                 std::string msg = "    ky = " + str(ky);
//                 for (int l = msg.length(); l < 44; l++) msg += " ";
//                 logger(LOG_SHOWCHARVAL) << msg;
//             }
//         }
//     } else { // target == THRESHOLD
//         if (current_loglevel | LOG_SHOWCHARVAL) {
//             if (variable == FREQUENCY) {
//                 std::string msg = "    k0,gain = " + str(real(k0)) + "," + str(mgain) + " ";
//                 for (int l = msg.length(); l < 48; l++) msg += " ";
//                 logger(LOG_SHOWCHARVAL) << msg;
//             } else if (variable == WAVEVECTOR_X) {
//                 std::string msg = "    kx,gain = " + str(real(kx)) + "," + str(mgain) + " ";
//                 for (int l = msg.length(); l < 48; l++) msg += " ";
//                 logger(LOG_SHOWCHARVAL) << msg;
//             } else { // variable == WAVEVECTOR_Y
//                 std::string msg = "    ky,gain = " + str(real(ky)) + "," + str(mgain) + " ";
//                 for (int l = msg.length(); l < 48; l++) msg += " ";
//                 logger(LOG_SHOWCHARVAL) << msg;
//             }
//         }
//     }
//
//     // We change the matrices M and A so we will have to find the new fields
//     fields_determined = false;
//
//     K0 = k0; Kx = kx; Ky = ky; MGain = mgain;
//
//     diagonalizer.initDiagonalization(k0, Kx, Ky, MGain);
//
//     // Obtain admittance
//     getFinalMatrix();
//
//     int N = M.rows();
//
//     // This is probably expensive but necessary check to avoid hangs
//     int NN = N*N;
//     for (int i = 0; i < NN; i++) {
//         if (isnan(real(M.data[i])) || isnan(imag(M.data[i])))
//             throw "SlabSolverBase::char_val: NaN in admittance matrix";
//     }
//
//     // Find the eigenvalues of M using LAPACK
//     dcomplex nth; int info;
//     F(zgeev)('N', 'N', N, M.data, N, evals, &nth, 1, &nth, 1, work, lwork, rwork, info);
//     if (info != 0) throw "SlabSolverBase::char_val: zgeev failed";
//
//     // Find the smallest eigenvalue
//     dcomplex val, result;
//     double min_mag = 1e32;
//     for (int i = 0; i < N; i++) {
//         val = evals[i];
//         if (abs(val) < min_mag) { min_mag = abs(val); result = val; }
//     }
// //     // Find the determinant
// //     dcomplex result = 1.;
// //     for (int i = 0; i < N; i++) {
// //         result *= evals[i];
// //     }
// //     result = log(result);
//
//     logger(LOG_SHOWCHARVAL) << "char_val = " << str(abs(result), "%.9g") << "\n";
//
//     // Field not found
//     interface_field = NULL;
//
//     return result;
// }
//
//
// //******************************************************************************
// /// Find the electric field at the interface after findig the solution
// /**  The field calculation is correct assuming that last call to char_var returned
//      the value close enough to 0 (less than 1e-10 or something)
//      TODO: Add error handling here (check for char_val etc.) */
// cvector SlabSolverBase::getInterfaceFieldVectorE(const dcomplex k0, const dcomplex kx, const dcomplex ky, double mgain)
// {
//     if (k0 != K0 || kx != Kx || ky != Ky || mgain != MGain) {
//         K0 = k0; Kx = kx; Ky = ky; MGain = mgain;
//         diagonalizer.initDiagonalization(k0, kx, ky, mgain);
//         interface_field = NULL;
//     }
//
//     int N = M.rows();
//
//     // Check if the necessary memory is already allocated
//     if (interface_field_matrix.rows() != N) {
//         interface_field_matrix = cmatrix(N,N);
//         interface_field = NULL;
//     }
//
//     // If the field already found, don't compute again
//     if (!interface_field) {
//
//         // Obtain admittance
//         getFinalMatrix();
//
//         // Find the eigenvalues of M using LAPACK
//         dcomplex nth; int info;
//         F(zgeev)('N', 'V', N, M.data, N, evals, &nth, 1, interface_field_matrix.data, N, work, lwork, rwork, info);
//         if (info != 0) throw "SlabSolverBase::getInterfaceFieldVectorE: zgeev failed";
//
//         // Find the number of the smallest eigenvalue
//         double mag, min_mag = 1e32;
//         int n;
//         for (int i = 0; i < N; i++) {
//             mag = abs(evals[i]);
//             if (mag < min_mag) { min_mag = mag; n = i; }
//         }
//
//         // Chose the eigenvector corresponding to the smallest eigenvalue
//         interface_field = interface_field_matrix.data + n*N;
//     }
//
//     // Make a copy of the interface vector
//     cvector E(N);
//     for (int i = 0; i < N; i++) E[i] = interface_field[i];
//
//     return E;
// }
//
// //******************************************************************************
// // Return the electic field components on a given grid
// void SlabSolverBase::interfaceFieldE(const std::vector<double> X, const std::vector<double> Y,
//                                     dcomplex* outVectorField2D)
// {
//     int layer = stack[interface-1].indx;
//
//     int N = diagonalizer.matrixSize();
//     int N0 = diagonalizer.getGrid().matrixSize();
//
//     cvector tmp1(N), E, tmp2(N), H(N0);
//     E = getInterfaceFieldVectorE(K0, Kx, Ky);
//
//     // H = TH * A * invTE * E
//     mult_matrix_by_vector(diagonalizer.invTE(stack[interface-1].indx), E, tmp1);
//     mult_matrix_by_vector(A, tmp1, tmp2);
//     mult_matrix_by_vector(diagonalizer.TH(stack[interface-1].indx), tmp2, H);
//
//     diagonalizer.getGrid().fieldE(layer, X, Y, K0, Kx, Ky, E, H, outVectorField2D);
// }
//
// //------------------------------------------------------------------------------
// // Return the magnetic field components on a given grid
// void SlabSolverBase::interfaceFieldH(const std::vector<double> X, const std::vector<double> Y,
//                                     dcomplex* outVectorField2D)
// {
//     int layer = stack[interface-1].indx;
//
//     int N = diagonalizer.matrixSize();
//     int N0 = diagonalizer.getGrid().matrixSize();
//
//     cvector tmp1(N), E, tmp2(N), H(N0);
//     E = getInterfaceFieldVectorE(K0, Kx, Ky);
//
//     // H = TH * A * invTE * E
//     mult_matrix_by_vector(diagonalizer.invTE(stack[interface-1].indx), E, tmp1);
//     mult_matrix_by_vector(A, tmp1, tmp2);
//     mult_matrix_by_vector(diagonalizer.TH(stack[interface-1].indx), tmp2, H);
//
//     diagonalizer.getGrid().fieldH(layer, X, Y, K0, Kx, Ky, E, H, outVectorField2D);
// }
//
// //******************************************************************************
// // Return the electic field components on a given grid
// void SlabSolverBase::fieldE(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = stack[n].indx;
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E = getFieldVectorE(z, n), H = getFieldVectorH(z, n);
//         if (n >= interface || (interface==count && Z[i]>0))
//             { int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j]; }
//         diagonalizer.getGrid().fieldE(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //------------------------------------------------------------------------------
// // Return the magnetic field components on a given grid
// void SlabSolverBase::fieldH(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = stack[n].indx;
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E = getFieldVectorE(z, n), H = getFieldVectorH(z, n);
//         if (n >= interface || (interface==count && Z[i]>0))
//             { int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j]; }
//         diagonalizer.getGrid().fieldH(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //******************************************************************************
// // Return the electric field components on a given grid
// void SlabSolverBase::backwardFieldE(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = stack[n].indx;
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//         }
//         diagonalizer.getGrid().fieldE(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //------------------------------------------------------------------------------
// // Return the magnetic field components on a given grid
// void SlabSolverBase::backwardFieldH(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = stack[n].indx;
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//         }
//         diagonalizer.getGrid().fieldH(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //******************************************************************************
//
// //******************************************************************************
// // Return the electric field components on a given grid
// void SlabSolverBase::forwardFieldE(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = stack[n].indx;
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//         }
//         diagonalizer.getGrid().fieldE(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //------------------------------------------------------------------------------
// // Return the magnetic field components on a given grid
// void SlabSolverBase::forwardFieldH(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = stack[n].indx;
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//         }
//         diagonalizer.getGrid().fieldH(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// /// Get the number of the layer containing given z
// int SlabSolverBase::layer_of_z(double& z)
// {
//     // reflect z for symmetric structure if necessary
//     if (interface == count) {
//         if (z > 0) z = -z;
//     }
//
//     int start, end, step;
//     if (z > 0) { start = interface; end = count; step = 1; }
//     else { start = interface-1; end = -1; step = -1; z = -z; }
//
//     double Z = 0, d;
//     for (int i = start; i != end; i+=step) {
//         d = stack[i].height;
//         if (Z <= z && z <= Z+d) {
//             z -= Z;
//             return i;
//         }
//         Z += d;
//     }
//
//     if (d == 0.) {
//         z -= Z;
//         return end-step;
//     } else if (z <= 1.01 * Z) { // 1% error margin
//         z = d;
//         return count-1;
//     } else {
//         throw "SlabSolverBase::layer_of_z(double z): z out of the structure";
//     }
// }

//     /// Store P matrix if we want it for field compuation
//     virtual void storeP(int n) {
//         if (allP) {
//             int N = diagonalizer.matrixSize();
//             if (memP.size() != count) {
//                 // Allocate the storage for admittance matrices
//                 memP.resize(count);
//                 for (int i = 0; i < count; i++) memP[i] = cmatrix(N,N);
//             }
//             memcpy(memP[n].data, P.data, N*N*sizeof(dcomplex));
//         }
//     }

// //**************************************************************************
// #define ee A
// #define hh P
//
// //**************************************************************************
// /// Coomon subroutine to compute the A and M matrices
// void ReflectionSolver::getFinalMatrix()
// {
//     // If the structure is not symmetric, consider another part
//     if (interface != count) {
//         getAM(count-1, interface, false);
//         getAM(0, interface-1, true);
//     } else {
//         getAM(0, interface-1, false, 2.);
//     }
// }
//
// //**************************************************************************
// /// Function to get A and M matrices for the half of the structure
// void ReflectionSolver::getAM(int start, int end, bool add, double mfac)
// {
//     // Get matrices sizes
//     int N0 = diagonalizer.getGrid().matrixSize();
//     int N = diagonalizer.matrixSize(); // <= N0
//     int NN = N*N;
//     cmatrix wrk(N, N0, work);    // matrix object for the workspace
//
//     findReflection(start, end);
//
//     cdiagonal gamma = diagonalizer.Gamma(stack[end].indx);
//     double H = stack[end].height;
//     for (int i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);
//
//     mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P := phas * P * phas
//     memcpy(A.data, P.data, NN*sizeof(dcomplex));                                // A := P
//
//     // A = [ phas*P*phas - I ] [ phas*P*phas + I ]^{-1}
//     for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P.data[ii] += 1;           // P := P + I
//     for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) A.data[ii] -= 1;           // A := A - I
//     int info;
//     F(zgetrf)(N, N, P.data, N, ipiv, info);                                     // P := LU(P)
//     F(ztrsm)('R', 'U', 'N', 'N', N, N, 1., P.data, N, A.data, N);               // A := A * U^{-1}
//     F(ztrsm)('R', 'L', 'N', 'U', N, N, 1., P.data, N, A.data, N);               // A := A * L^{-1}
//     // reorder columns (there is no such function in LAPACK)
//     for (int j = N-1; j >=0 ; j--) {
//         int jp = ipiv[j]-1;
//         for (int i = 0; i < N; i++) {
//             register dcomplex t = A(i,j); A(i,j) = A(i,jp); A(i,jp) = t;
//         }
//     }
//
//     // M for the half of the structure
//     mult_matrix_by_matrix(A, diagonalizer.invTE(stack[end].indx), wrk);         // wrk := A * invTE[end]
//
//     F(zgemm)('N','N', N0, N0, N, mfac, diagonalizer.TH(stack[end].indx).data, N0,
//              work, N, add?1.:0., M.data, N0);                                   // M := mfac * TH[end] * wrk
// }
//
// //**************************************************************************
// /// Function to find reflection matrix for one half of the structure
// void ReflectionSolver::findReflection(int start, int end)
// {
//     // Should be called from 0 to interface-1
//     // and from count-1 to interface
//
//     const int inc = (start < end) ? 1 : -1;
//
//     int N0 = diagonalizer.getGrid().matrixSize();
//     int N = diagonalizer.matrixSize();
//     int NN = N*N;
//
//     cmatrix wrk(N, N0, work);    // matrix object for the workspace
//
//     cdiagonal gamma;
//
//     // in the beginning the P matrix is zero
//     memset(P.data, 0, NN*sizeof(dcomplex));
//     storeP(start);
//
//     diagonalizer.diagonalizeLayer(stack[start].indx);
//
//     for (int n = start; n != end; n += inc)
//     {
//         gamma = diagonalizer.Gamma(stack[n].indx);
//
//         double H = stack[n].height;
//         if (n != start) {
//             for (int i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);
//         } else {
//             for (int i = 0; i < N; i++) {
//                 dcomplex g = gamma[i];
//                 if (real(g) < -SMALL) g = -g;
//                 phas[i] = exp(-I*g*H);
//             }
//         }
//         mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);     // P := phas * P * phas
//
//         diagonalizer.diagonalizeLayer(stack[n+inc].indx);
//
//         // ee = invTE(n+1)*TE(n) * [ phas*P*phas + I ]
//         for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P.data[ii] += 1.;      // P := P.orig + I
//         mult_matrix_by_matrix(diagonalizer.TE(stack[n].indx), P, wrk);          // wrk := TE[n] * P
//         mult_matrix_by_matrix(diagonalizer.invTE(stack[n+inc].indx), wrk, ee);  // ee := invTE[n+1] * wrk (= A)
//
//         // hh = invTH(n+1)*TH(n) * [ phas*P*phas - I ]
//         for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P.data[ii] -= 2.;      // P := P - I
//         // multiply rows of P by -1 where necessary for properly outgoing wave
//         if (n == start) {
//             for (int i = 0; i < N; i++)
//                 if (real(gamma[i]) < -SMALL)
//                     for(int j = 0; j < N; j++) P(i,j) = -P(i,j);
//         }
//         mult_matrix_by_matrix(diagonalizer.TH(stack[n].indx), P, wrk);          // wrk := TH[n] * P
//         mult_matrix_by_matrix(diagonalizer.invTH(stack[n+inc].indx), wrk, hh);  // hh := invTH[n+1] * wrk (= P)
//
//         // ee := ee-hh, hh := ee+hh
//         for (int i = 0; i < NN; i++) {
//             dcomplex e = ee.data[i], h = hh.data[i];
//             A.data[i] = e - h;
//             P.data[i] = e + h;
//         }
//
//         // There might appear a problem if the last layer and the one next to the last are the same.
//         // It may happen that e ~ h and then A matrix is close to singular. This will yield unreliable results.
//         // To avoid this never ever put two identical layers next to each other
//
//         // P = P * inv(A)
//         int info;
//         F(zgetrf)(N, N, A.data, N, ipiv, info);                                     // A := LU(A)         (= A)
//         if (info > 0) throw "ReflectionSolver::findReflection: Matrix [e(n) - h(n)] is singular";
//         F(ztrsm)('R', 'U', 'N', 'N', N, N, 1., A.data, N, P.data, N);               // P := P * U^{-1}    (= P)
//         F(ztrsm)('R', 'L', 'N', 'U', N, N, 1., A.data, N, P.data, N);               // P := P * L^{-1}
//         // reorder columns (there is no such function in LAPACK)
//         for (int j = N-1; j >=0 ; j--) {
//             int jp = ipiv[j]-1;
//             for (int i = 0; i < N; i++) {
//                 register dcomplex t = P(i,j); P(i,j) = P(i,jp); P(i,jp) = t;
//             }
//         }
//
//         storeP(n+inc);
//     }
// }
//
//
//
// //**************************************************************************
// /// Determine the diagonalized F and B fields in each layer
// void ReflectionSolver::determineFields(const dcomplex k0, const dcomplex kx, const dcomplex ky, const double mgain)
// {
//     logger(LOG_BASIC) << "ReflectionSolver: Determining fields...\n\n";
//
//     int N = diagonalizer.matrixSize();
//     int N0 = diagonalizer.getGrid().matrixSize();
//     int NN = N*N;
//
//     cdiagonal gamma;
//
//     // Assign the space for the field vectors
//     layerFields.resize(count);
//     // Obtain the physical fields at the last layer
//     allP = true; resetInterfaceField();
//
//     cvector E = getInterfaceFieldVectorE(k0, kx, ky, mgain);
//     cvector tmp(N, work);
//
//     for (int pass = 0; pass < 1 || (pass < 2 && interface != count); pass++)
//     {
//         // each pass for below and above the interface
//
//         int start, end, inc;
//         switch (pass)
//         {
//             case 0: start = interface-1; end = 0;              inc = -1; break;
//             case 1: start = interface;   end = count-1; inc = +1; break;
//         }
//
//         layerFields[start].F = cvector(N);
//         layerFields[start].B = cvector(N);
//
//         // compute B-field for the layer next to the interface
//         int curr = stack[start].indx;
//
//         gamma = diagonalizer.Gamma(curr);
//         double H = stack[start].height;
//         for (int i = 0; i < N; i++)
//             phas[i] = exp(-I*gamma[i]*H);
//
//         // P = phas*P*phas + I
//         memcpy(P.data, getP(start).data, NN*sizeof(dcomplex));
//         mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);             // P := phas * P * phas
//         for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P.data[ii] += 1.;              // P := P + I
//
//         mult_matrix_by_vector(diagonalizer.invTE(curr), E, layerFields[start].B);       // B := invTE * E
//         invmult(P, layerFields[start].B);                                               // B := inv(P) * B
//         for (int i = 0; i < N; i++) layerFields[start].B[i] *= phas[i];                 // B := phas * B
//
//         for (int n = start; n != end; n += inc)
//         {
//             // F-field for the current layer
//             mult_matrix_by_vector(getP(n), layerFields[n].B, layerFields[n].F);
//
//             // Compute B-field for the next (previous) layer
//
//             layerFields[n+inc].F = cvector(N);
//             layerFields[n+inc].B = cvector(N);
//
//             // some aliases
//             cvector& F1 = layerFields[n].F;
//             cvector& B1 = layerFields[n].B;
//             cvector& F2 = layerFields[n+inc].F;
//             cvector& B2 = layerFields[n+inc].B;
//
//             curr = stack[n].indx;
//             int next = stack[n+inc].indx;
//
//             gamma = diagonalizer.Gamma(next);
//
//             for (int i = 0; i < N; i++) F2[i] = F1[i] - B1[i];          // F2 := F1 - B1
//             mult_matrix_by_vector(diagonalizer.TH(curr), F2, tmp);      // tmp := TH * F2
//             mult_matrix_by_vector(diagonalizer.invTH(next), tmp, B2);   // B2 := invTH * E
//             // multiply rows of invTH by -1 where necessary for the outer layer
//             if (n+inc == end) {
//                 for (int i = 0; i < N; i++)
//                     if (real(gamma[i]) < SMALL) B2[i] = -B2[i];
//             }
//
//             for (int i = 0; i < N; i++) F2[i] = F1[i] + B1[i];          // F2 := F1 + B1
//             mult_matrix_by_vector(diagonalizer.TE(curr), F2, tmp);      // tmp := TE * F2
//             F(zgemm)('N','N', N, 1, N0, 1., diagonalizer.invTE(next).data, N,
//                      tmp.data, N0, -1., B2.data, N);                    // B2 := invTE * tmp - B2
//
//             H = stack[n+inc].height;
//             if (n+inc != end) {
//                 for (int i = 0; i < N; i++)
//                     B2[i] *= 0.5 * exp(-I*gamma[i]*H);                  // B2 := 1/2 * phas * B2
//             } else {
//                 for (int i = 0; i < N; i++) {
//                     dcomplex g = gamma[i];
//                     if (real(g) < SMALL) g = -g;
//                     B2[i] *= 0.5 * exp(-I*g*H);                         // B2 := 1/2 * phas * B2
//                 }
//             }
//         }
//
//         //mult_matrix_by_vector(getP(end), layerFields[end].B, layerFields[end].F);
//         for (int i = 0; i < N; ++i) layerFields[end].F[i] = 0.;
//
//         // In the outer layers replace F and B where necessary for consistent gamma handling
//         cvector& F2 = layerFields[end].F;
//         cvector& B2 = layerFields[end].B;
//         for (int i = 0; i < N; i++) {
//             if (real(gamma[i]) < -SMALL) {
//                 dcomplex t = B2[i]; B2[i] =  F2[i]; F2[i] = t;
//             }
//         }
//     }
//
//     allP = false;
//     fields_determined = true;
// }
//
//
// //**************************************************************************
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getFieldVectorE(double z, int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::getFieldVectorE: Fields not determined";
//
//     cvector& FF = layerFields[n].F;
//     cvector& BB = layerFields[n].B;
//
//     cdiagonal gamma = diagonalizer.Gamma(stack[n].indx);
//
//     int N = gamma.size();
//     cvector E(N);
//
//     z = stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         E[i] = FF[i] * exp(phi) + BB[i] * exp(-phi);
//     }
//
//     return diagonalizer.TE(stack[n].indx) * E;
// }
//
// //--------------------------------------------------------------------------
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getFieldVectorH(double z, int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::getFieldVectorE: Fields not determined";
//
//     cvector& FF = layerFields[n].F;
//     cvector& BB = layerFields[n].B;
//
//     cdiagonal gamma = diagonalizer.Gamma(stack[n].indx);
//
//     int N = gamma.size();
//     cvector H(N);
//
//     z = stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         H[i] = (FF[i] * exp(phi) - BB[i] * exp(-phi));
//     }
//
//
//     if (n == 0 || (interface != count && n == count-1)) {
//         cdiagonal gamma = diagonalizer.Gamma(stack[n].indx);
//         // In the outer layers multiply H by -1 where necessary for propagating wave
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) H[i] = - H[i];
//     }
//
//     return diagonalizer.TH(stack[n].indx) * H;
// }
//
// cvector ReflectionSolver::getBackwardFieldVectorE(double z, int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::getBackwardFieldVectorE: Fields not determined";
//
//     cdiagonal gamma = diagonalizer.Gamma(stack[n].indx);
//     int N = gamma.size();
//
//     cvector FF = layerFields[n].F.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) FF[i] = layerFields[n].B[i];
//
//     cvector E(N);
//
//     z = stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         E[i] = FF[i] * exp(phi);
//     }
//
//     return diagonalizer.TE(stack[n].indx) * E;
// }
//
// //**************************************************************************
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getForwardFieldVectorE(double z, int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::getForwardFieldVectorE: Fields not determined";
//
//     cdiagonal gamma = diagonalizer.Gamma(stack[n].indx);
//     int N = gamma.size();
//
//     cvector BB = layerFields[n].B.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) BB[i] = layerFields[n].F[i];
//
//     cvector E(N);
//
//     z = stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         E[i] = BB[i] * exp(-phi);
//     }
//
//     return diagonalizer.TE(stack[n].indx) * E;
// }
//
// //--------------------------------------------------------------------------
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getBackwardFieldVectorH(double z, int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::getBackwardFieldVectorH: Fields not determined";
//
//     cdiagonal gamma = diagonalizer.Gamma(stack[n].indx);
//     int N = gamma.size();
//
//     cvector FF = layerFields[n].F.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) FF[i] = layerFields[n].B[i];
//
//     cvector H(N);
//
//     z = stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         H[i] = FF[i] * exp(phi);
//     }
//
//     return diagonalizer.TH(stack[n].indx) * H;
// }
//
//
// //--------------------------------------------------------------------------
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getForwardFieldVectorH(double z, int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::getForwardFieldVectorH: Fields not determined";
//
//     cdiagonal gamma = diagonalizer.Gamma(stack[n].indx);
//     int N = gamma.size();
//
//     cvector BB = layerFields[n].B.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) BB[i] = layerFields[n].F[i];
//
//     cvector H(N);
//
//     z = stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         H[i] = - BB[i] * exp(-phi);
//     }
//
//     return diagonalizer.TH(stack[n].indx) * H;
// }
//
//
//
// //**************************************************************************
// /// Return the intensity of the backward propagating field for a given layer
// double ReflectionSolver::backwardEIntegral(int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::backwardFieldIntegral: Fields not determined";
//
//     int layer = stack[n].indx;
//
//     cdiagonal gamma = diagonalizer.Gamma(layer);
//     int N = diagonalizer.matrixSize();
//     int N0 = diagonalizer.getGrid().matrixSize();
//     double H = stack[n].height;
//
//     cvector E(N0), F;
//
//     if (n < interface) {
//         F = layerFields[n].F.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) F[i] = layerFields[n].B[i];
//     } else {
//         F = layerFields[n].B.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) F[i] = layerFields[n].F[i];
//     }
//
//     mult_matrix_by_vector(diagonalizer.TE(layer), F, E);
//
//     double integral = 0.;
//
//     for (int i = 0; i < N0; i++) {
//         integral += 0.5 * real(conj(E[i])* E[i]);
//     }
//
//     return integral;
// }
//
// //--------------------------------------------------------------------------
// /// Return the intensity of the forward propagating field for a given layer
// double ReflectionSolver::forwardEIntegral(int n)
// {
//     if (!fields_determined)
//         throw "ReflectionSolver::forwardFieldIntegral: Fields not determined";
//
//     int layer = stack[n].indx;
//
//     cdiagonal gamma = diagonalizer.Gamma(layer);
//     int N = diagonalizer.matrixSize();
//     int N0 = diagonalizer.getGrid().matrixSize();
//     double H = stack[n].height;
//
//     cvector E(N0), B;
//
//     if (n < interface) {
//         B = layerFields[n].B.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) B[i] = layerFields[n].F[i];
//     } else {
//         B = layerFields[n].F.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) B[i] = layerFields[n].B[i];
//     }
//
//     mult_matrix_by_vector(diagonalizer.TE(layer), B, E);
//
//     double integral = 0.;
//
//     for (int i = 0; i < N0; i++) {
//         integral += 0.5 * real(conj(E[i])* E[i]);
//     }
//
//     return integral;
// }
//
// //**************************************************************************
// /// Function to get reflected field
// cvector ReflectionSolver::reflectedAmplitudes(dcomplex k0, cvector Ei)
// {
//     // Get reflection matrix
//     diagonalizer.initDiagonalization(k0, Kx, Ky, MGain);
//     findReflection(0, count-1);
//
//     int layer = stack[count-1].indx;
//
//     // Ei is an incident wave
//     cvector B = diagonalizer.invTE(layer) * Ei; // diagonalized incident E-field
//
//     cvector F = P * B;                          // diagonalized reflected E-field
//     cvector Er = diagonalizer.TE(layer) * F;    // reflected E-field
//
//     return Er;
// }
//
// /// Function to determine field on reflection in the whole structure
// void ReflectionSolver::determineReflectedFields(dcomplex k0, cvector Ei)
// {
//     logger(LOG_BASIC) << "ReflectionSolver: Determining reflected fields...\n\n";
//
//     // Assign the space for the field vectors
//     layerFields.resize(count);
//     // Obtain the physical fields at the last layer
//     allP = true; K0 = k0;
//
//     // Compute reflection matrices
//     diagonalizer.initDiagonalization(k0, Kx, Ky, MGain);
//     findReflection(0, count-1);
//
//     // Temporary and initial data
//     int N = diagonalizer.matrixSize();
//     int N0 = diagonalizer.getGrid().matrixSize();
//     cvector tmp(N, work);
//     cdiagonal gamma;
//
//     int curr = stack[0].indx;
//     double H;
//
//     layerFields[count-1].B = diagonalizer.invTE(curr) * Ei; // diagonalized incident E-field
//     layerFields[count-1].F = cvector(N);
//
//     for (int n = count-1; n > 0; n--)
//     {
//         // F-field for the current layer
//         mult_matrix_by_vector(getP(n), layerFields[n].B, layerFields[n].F);
//
//         // Compute B-field for the next (previous) layer
//
//         layerFields[n-1].F = cvector(N);
//         layerFields[n-1].B = cvector(N);
//
//         // some aliases
//         cvector& F1 = layerFields[n].F;
//         cvector& B1 = layerFields[n].B;
//         cvector& F2 = layerFields[n-1].F;
//         cvector& B2 = layerFields[n-1].B;
//
//         curr = stack[n].indx;
//         int next = stack[n-1].indx;
//
//         gamma = diagonalizer.Gamma(next);
//
//         for (int i = 0; i < N; i++) F2[i] = F1[i] - B1[i];          // F2 := F1 - B1
//         mult_matrix_by_vector(diagonalizer.TH(curr), F2, tmp);      // tmp := TH * F2
//         mult_matrix_by_vector(diagonalizer.invTH(next), tmp, B2);   // B2 := invTH * E
//         // multiply rows of invTH by -1 where necessary for the outer layer
//         if (n == 1) {
//             for (int i = 0; i < N; i++)
//                 if (real(gamma[i]) < -SMALL) B2[i] = -B2[i];
//         }
//
//         for (int i = 0; i < N; i++) F2[i] = F1[i] + B1[i];          // F2 := F1 + B1
//         mult_matrix_by_vector(diagonalizer.TE(curr), F2, tmp);      // tmp := TE * F2
//         F(zgemm)('N','N', N, 1, N0, 1., diagonalizer.invTE(next).data, N,
//                     tmp.data, N0, -1., B2.data, N);                 // B2 := invTE * tmp - B2
//
//         H = stack[n-1].height;
//         if (n != 1) {
//             for (int i = 0; i < N; i++)
//                 B2[i] *= 0.5 * exp(-I*gamma[i]*H);                  // B2 := 1/2 * phas * B2
//         } else {
//             for (int i = 0; i < N; i++) {
//                 dcomplex g = gamma[i];
//                 if (real(g) < -SMALL) g = -g;
//                 B2[i] *= 0.5 * exp(-I*g*H);                         // B2 := 1/2 * phas * B2
//             }
//         }
//     }
//
//     //mult_matrix_by_vector(getP(0), layerFields[0].B, layerFields[0].F);
//     layerFields[0].F = cvector(N);
//
//     // In the outer layers replace F and B where necessary for consistent gamma handling
//     for (int n = 0; n < count; n += count-1) {
//         cvector& F2 = layerFields[n].F;
//         cvector& B2 = layerFields[n].B;
//         gamma = diagonalizer.Gamma(stack[n].indx);
//         for (int i = 0; i < N; i++) {
//             if (real(gamma[i]) < -SMALL) {
//                 dcomplex t = B2[i]; B2[i] =  F2[i]; F2[i] = t;
//             }
//         }
//      }
//
//     // Replace F and B above the interface for consistency in getFieldVectorE and getFieldVectorH
//     for (int n = interface; n < count; n++) {
//         cvector& F2 = layerFields[n].F;
//         cvector& B2 = layerFields[n].B;
//         gamma = diagonalizer.Gamma(stack[n].indx);
//         H = stack[n].height;
//         for (int i = 0; i < N; i++) {
//                 dcomplex phas = exp(-I*gamma[i]*H);
//                 dcomplex t = B2[i] / phas;
//                 B2[i] =  F2[i] * phas;
//                 F2[i] = t;
//         }
//     }
//
//     // for (int n = count-1; n >= 0; n--) {
//     //     //std::cerr << "g[" << n << "]: ";
//     //     for (int i = 0; i < N; i++) {
//     //         double gr = real(diagonalizer.Gamma(stack[n].indx)[i]);
//     //         double gi = imag(diagonalizer.Gamma(stack[n].indx)[i]);
//     //         //std::cerr << ((abs(gr)>1e-10)?gr:0.) << ((gi>=0)?"+":"-") << ((abs(gi)>1e-10)?abs(gi):0.) << "j ";
//     //     }
//     //     std::cerr << "\nF[" << n << "]: ";
//     //     for (int i = 0; i < N; i++) std::cerr << ((abs(layerFields[n].F[i])>1e-10)?abs(layerFields[n].F[i]):0.) << " ";
//     //     std::cerr << "\nB[" << n << "]: ";
//     //     for (int i = 0; i < N; i++) std::cerr << ((abs(layerFields[n].B[i])>1e-10)?abs(layerFields[n].B[i]):0.) << " ";
//     //     std::cerr << "\n\n";
//     // }
//
//     allP = false;
//     fields_determined = true;
// }

template class ReflectionSolver<Geometry2DCartesian>;
template class ReflectionSolver<Geometry2DCylindrical>;
template class ReflectionSolver<Geometry3D>;

}}} // namespace plask::solvers::slab
