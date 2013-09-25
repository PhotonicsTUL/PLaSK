#include "expansion_pw2d.h"
#include "reflection_solver_2d.h"

namespace plask { namespace solvers { namespace slab {

void ExpansionPW2D::init(bool long_zero, bool tran_zero)
{
    auto geometry = solver->getGeometry();

    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    left = geometry->getChild()->getBoundingBox().lower[0];
    right = geometry->getChild()->getBoundingBox().upper[0];

    size_t refine = solver->refine, M;
    if (refine == 0) refine = 1;

    if (long_zero || tran_zero) {
        // Test for off-diagonal NR components in which case we cannot use neither symmetry nor separation
        bool off_diagonal = false;
        double dx =  (right-left) / (2 * solver->getSize() * refine);
        for (const RectilinearAxis& axis1: solver->getLayersPoints()) {
            for (double x = left+dx/2; x < right; x += dx) {
                Tensor3<dcomplex> nr = geometry->getMaterial(vec(x,axis1[0]))->NR(real(solver->getWavelength()), 300.);
                if (nr.c01 != 0. || nr.c10 != 0.) { off_diagonal = true; break; }
            }
            if (off_diagonal) break;
        }
        if (off_diagonal) {
            symmetric = separated = false;
        } else {
            symmetric = geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN) && tran_zero;
            separated = long_zero;
        }
    } else {
        symmetric = geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN) && tran_zero;
        separated = long_zero;
    }

    if (geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN)) {
        if (right <= 0) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0) throw BadMesh(solver->getId(), "Symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric) left = -right;
    }

    if (!periodic) {
        // Add PMLs
        if (!symmetric) left -= solver->pml.size + solver->pml.shift;
        right += solver->pml.size + solver->pml.shift;
    }

    double L;
                                                            // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric) {                                       //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        L = right - left;                                   //  ^ ^ ^ ^ ^
        N = 2 * solver->getSize() + 1;                      // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nN = 4 * solver->getSize() + 1;
        M = refine * nN;                                    // N = 3  nN = 5  refine = 4  M = 20
        double dx = 0.5 * L * (refine-1) / M;               // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        xmesh = RegularAxis(left-dx, right-dx-L/M, M);      //  ^ ^ ^ ^
        xpoints = RegularAxis(left, right-L/N, N);          // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    } else {
        L = 2 * right;
        N = solver->getSize() + 1;
        nN = 2 * solver->getSize() + 1;                     // N = 3  nN = 5  refine = 4  M = 20
        M = refine * nN;                                    // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        double dx = 0.25 * L / M;                           //  ^ ^ ^ ^
        xmesh = RegularAxis(left + dx, right - dx, M);      // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        dx = 0.25 * L / N;
        xpoints = RegularAxis(left + dx, right - dx, N);
    }

    solver->writelog(LOG_DETAIL, "Creating expansion with %1% plane-waves (matrix size: %2%)", N, matrixSize());

    coeffs.reset(nN);
    matFFT = FFT::Forward1D(5, nN, symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE, reinterpret_cast<dcomplex*>(coeffs.data()));

    // Compute permeability coefficients
    mag.reset(nN, Tensor2<dcomplex>(0.));
    if (periodic) {
        mag[0].c00 = 1.; mag[0].c11 = 1.; // constant 1
    } else {
        DataVector<double> Sy(M, 0.);   // PML coeffs for mu
        // Add PMLs
        solver->writelog(LOG_DETAIL, "Adding side PMLs (total structure width: %1%um)", L);
        double pl = left + solver->pml.size, pr = right - solver->pml.size;
        if (symmetric) pil = 0;
        else pil = std::lower_bound(xmesh.begin(), xmesh.end(), pl) - xmesh.begin();
        pir = std::lower_bound(xmesh.begin(), xmesh.end(), pr) - xmesh.begin();
        for (size_t i = 0; i != pil; ++i) {
            double h = (pl - xmesh[i]) / solver->pml.size;
            Sy[i] = solver->pml.extinction * pow(h, solver->pml.order);
        }
        for (size_t i = pir+1; i != xmesh.size(); ++i) {
            double h = (xmesh[i] - pr) / solver->pml.size;
            Sy[i] = solver->pml.extinction * pow(h, solver->pml.order);
            dcomplex sy(1., Sy[i]);
        }
        // Average mu
        std::fill(mag.begin(), mag.end(), Tensor2<dcomplex>(0.));
        for (size_t i = 0; i != nN; ++i) {
            for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
                dcomplex sy(1., Sy[j]);
                mag[i] += Tensor2<dcomplex>(sy, 1./sy);
            }
            mag[i] /= refine;
        }
        // Compute FFT
        FFT::Forward1D(2, nN, symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE, reinterpret_cast<dcomplex*>(mag.data())).execute();
        // Smooth coefficients
        if (solver->smooth) {
            double bb4 = M_PI / L; bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN; ++i) {
                int k = i; if (k > nN/2) k -= nN;
                mag[i] *= exp(-solver->smooth * bb4 * k * k);
            }
        }
    }
}


size_t ExpansionPW2D::lcount() const {
    return solver->getLayersPoints().size();
}


bool ExpansionPW2D::diagonalQE(size_t l) const {
    return false;
}


void ExpansionPW2D::getMaterialCoefficients(size_t l)
{
    if (isnan(real(solver->getWavelength())) || isnan(imag(solver->getWavelength())))
        throw BadInput(solver->getId(), "No wavelength set in solver");

    auto geometry = solver->getGeometry();
    const RectilinearAxis& axis1 = solver->getLayerPoints(l);

    size_t refine = solver->refine;
    size_t M = refine * nN;

    solver->writelog(LOG_DETAIL, "Getting refractive indices for layer %1% (sampled at %2% points)", l, M);

    DataVector<Tensor3<dcomplex>> NR(M);

    RectilinearMesh2D mesh(xmesh, axis1, RectilinearMesh2D::ORDER_TRANSPOSED);
    auto temperature = solver->inTemperature(mesh);
    double lambda = real(solver->getWavelength());
    double maty = axis1[0]; // at each point along any vertical axis material is the same
    for (size_t i = 0; i < M; ++i) {
        auto material = geometry->getMaterial(Vec<2>(xmesh[i],maty));
        // assert([&]()->bool{for(auto y: axis1)if(geometry->getMaterial(Vec<2>(xmesh[i],y))!=material)return false; return true;}());
        double T = 0.; // average temperature in all vertical points
        for (size_t j = i * axis1.size(), end = (i+1) * axis1.size(); j != end; ++j) T += temperature[j];
        T /= axis1.size();
        NR[i] = material->NR(lambda, T);
    }

    for (Tensor3<dcomplex>& val: NR) val.sqr_inplace(); // make epsilon from NR

    // Add PMLs
    if (!periodic) {
        Tensor3<dcomplex> ref;
        double pl = left + solver->pml.size, pr = right - solver->pml.size;
        ref = NR[pil];
        for (size_t i = 0; i != pil; ++i) {
            double h = (pl - xmesh[i]) / solver->pml.size;
            dcomplex sy(1., solver->pml.extinction * pow(h, solver->pml.order));
            NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
        }
        ref = NR[pir];
        for (size_t i = pir+1; i != xmesh.size(); ++i) {
            double h = (xmesh[i] - pr) / solver->pml.size;
            dcomplex sy(1., solver->pml.extinction * pow(h, solver->pml.order));
            NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
        }
    }

    // Average material parameters
    coeffs.fill(Tensor3<dcomplex>(0.));
    double factor = 1. / refine;
    for (size_t i = 0; i != nN; ++i) {
        for (size_t j = refine*i, end = refine*(i+1); j != end; ++j)
            coeffs[i] += Tensor3<dcomplex>(NR[j].c00, 1./NR[j].c11, NR[j].c22, NR[j].c01, NR[j].c10);
        coeffs[i] *= factor;
        coeffs[i].c11 = 1. / coeffs[i].c11; // We were averaging inverses of c11 (xx)
        coeffs[i].c22 = 1. / coeffs[i].c22; // We need inverse of c22 (yy)
    }

    // Perform FFT
    matFFT.execute();

    // Smooth coefficients
    if (solver->smooth) {
        double bb4 = M_PI / ((right-left) * (symmetric? 2 : 1)); bb4 *= bb4;   // (2π/L)² / 4
        for (size_t i = 0; i != nN; ++i) {
            int k = i; if (k > nN/2) k -= nN;
            coeffs[i] *= exp(-solver->smooth * bb4 * k * k);
        }
    }

    // Cache coefficients required for field computations
    //TODO
}


DataVector<const Tensor3<dcomplex>> ExpansionPW2D::getMaterialNR(size_t l, const RectilinearAxis mesh,
                                                                InterpolationMethod interp)
{
    double L = right - left;
    getMaterialCoefficients(l);
    DataVector<Tensor3<dcomplex>> result;
    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        result.reset(mesh.size(), Tensor3<dcomplex>(0.));
        if (!symmetric) {
            for (int k = -int(nN)/2, end = int(nN+1)/2; k != end; ++k) {
                size_t j = (k>=0)? k : k + nN;
                for (size_t i = 0; i != mesh.size(); ++i) {
                    result[i] += coeffs[j] * exp(2*M_PI * k * I * (mesh[i]-left) / L);
                }
            }
        } else {
            for (size_t i = 0; i != mesh.size(); ++i) {
                result[i] += coeffs[0];
                for (int k = 1; k != nN; ++k) {
                    result[i] += 2. * coeffs[k] * cos(M_PI * k * mesh[i] / L);
                }
            }
        }
    } else {
        FFT::Backward1D(5, nN, symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE, reinterpret_cast<dcomplex*>(coeffs.data())).execute();
        RegularAxis cmesh;
        if (symmetric) {
            double dx = 0.5 * (right-left) / nN;
            cmesh.reset(left + dx, right - dx, nN);
        } else {
            cmesh.reset(left, right, nN+1);
            auto old = coeffs;
            coeffs.reset(nN+1);
            std::copy(old.begin(), old.end(), coeffs.begin());
            coeffs[old.size()] = old[0];
        }
        RegularMesh2D src_mesh(cmesh, RegularAxis(0,0,1));
        RectilinearMesh2D dst_mesh(mesh, RectilinearAxis({0}));
        result = interpolate(src_mesh, coeffs, WrappedMesh<2>(dst_mesh, solver->getGeometry()), interp);
    }
    for (Tensor3<dcomplex>& eps: result) {
        eps.c22 = 1. / eps.c22;
        eps.sqrt_inplace();
    }
    return result;
}



void ExpansionPW2D::getMatrices(size_t l, dcomplex k0, dcomplex beta, dcomplex kx, cmatrix& RE, cmatrix& RH)
{
    getMaterialCoefficients(l);
    
    int order = solver->getSize();
    dcomplex f = I / k0, k02 = k0*k0;
    dcomplex b = 2*M_PI / (right-left) * (symmetric? 0.5 : 1.0);

    if (separated) {
        //TODO
        if (symmetric) {
            // Separated symmetric
            std::fill_n(RE.data(), N*N, dcomplex(0.));
            std::fill_n(RH.data(), N*N, dcomplex(0.));
            if (polarization == TE) {
                for (int i = 0; i <= order; ++i) {
                    dcomplex gi = b * double(i) - kx;
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
                        dcomplex fz = (j < 0 && symmetry == SYMMETRIC_E_TRAN)? -f : f;
                        RH(iE(i), iH(j)) = - k02 * muxx(ij);
                        RE(iH(i), iE(j)) =   fz * gi * gj * imuyy(ij) - k02 * epszz(ij);
                    }
                }
            } else {
                for (int i = 0; i <= order; ++i) {
                    dcomplex gi = b * double(i) - kx;
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
                        dcomplex fx = (j < 0 && symmetry == SYMMETRIC_E_LONG)? -f : f;
                        RH(iE(i), iH(j)) = - fx * gi * gj * iepsyy(ij) + k02 * muzz(ij);
                        RE(iH(i), iE(j)) =   k02 * epsxx(ij);
                    }
                }
            }
        } else {
            // Separated asymmetric
            if (polarization == TE) {
                for (int i = -order; i <= order; ++i) {
                    dcomplex gi = b * double(i) - kx;
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
                        RH(iE(i), iH(j)) = - k02 * muxx(ij);
                        RE(iH(i), iE(j)) =   f * gi * gj * imuyy(ij) - k02 * epszz(ij);
                    }
                }
            } else {
                for (int i = -order; i <= order; ++i) {
                    dcomplex gi = b * double(i) - kx;
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
                        RH(iE(i), iH(j)) = - f * gi * gj * iepsyy(ij) + k02 * muzz(ij);
                        RE(iH(i), iE(j)) =   k02 * epsxx(ij);
                    }
                }
            }
        }
    } else {
        if (symmetric) {
            // Full symmetric
            std::fill_n(RE.data(), 4*N*N, dcomplex(0.));
            std::fill_n(RH.data(), 4*N*N, dcomplex(0.));
            for (int i = 0; i <= order; ++i) {
                dcomplex gi = b * double(i);
                for (int j = -order; j <= order; ++j) {
                    dcomplex gj = b * double(j);
                    int ij = abs(i-j);
                    dcomplex fx = (j < 0 && symmetry == SYMMETRIC_E_LONG)? -f : f;
                    dcomplex fz = (j < 0 && symmetry == SYMMETRIC_E_TRAN)? -f : f;
                    RH(iEx(i), iHz(j)) += - fx * gi * gj * iepsyy(ij) + k02 * muzz(ij);
                    RH(iEz(i), iHz(j)) +=   fx * beta * gj * iepsyy(ij);
                    RH(iEx(i), iHx(j)) += - fz * beta * gi * iepsyy(ij);
                    RH(iEz(i), iHx(j)) +=   fz * beta*beta * iepsyy(ij) - k02 * muxx(ij);
                    RE(iHz(i), iEx(j)) += - fx * beta*beta * imuyy(ij) + k02 * epsxx(ij);
                    RE(iHx(i), iEx(j)) +=   fx * beta * gi  * imuyy(ij);
                    RE(iHz(i), iEz(j)) += - fz * beta * gj * imuyy(ij);
                    RE(iHx(i), iEz(j)) +=   fz * gi * gj * imuyy(ij) + k02 * epszz(ij);
                }
            }
        } else {
            // Full asymmetric
            for (int i = -order; i <= order; ++i) {
                dcomplex gi = b * double(i) - kx;
                for (int j = -order; j <= order; ++j) {
                    int ij = i-j;   dcomplex gj = b * double(j) - kx;
                    RH(iEx(i), iHz(j)) = - f * gi * gj * iepsyy(ij) + k02 * muzz(ij);
                    RH(iEz(i), iHz(j)) =   f * beta * gj * iepsyy(ij);
                    RH(iEx(i), iHx(j)) = - f * beta * gi * iepsyy(ij);
                    RH(iEz(i), iHx(j)) =   f * beta*beta * iepsyy(ij) - k02 * muxx(ij);
                    RE(iHz(i), iEx(j)) = - f * beta*beta * imuyy(ij) + k02 * epsxx(ij);
                    RE(iHx(i), iEx(j)) =   f * beta * gi  * imuyy(ij) - k02 * epszx(ij);
                    RE(iHz(i), iEz(j)) = - f * beta * gj * imuyy(ij) + k02 * epsxz(ij);
                    RE(iHx(i), iEz(j)) =   f * gi * gj * imuyy(ij) - k02 * epszz(ij);
                }
            }
        }
    }
}




}}} // namespace plask
