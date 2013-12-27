#include "expansion_pw2d.h"
#include "fourier_reflection_2d.h"
#include "mesh_adapter.h"

#define SOLVER static_cast<FourierReflection2D*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionPW2D::ExpansionPW2D(FourierReflection2D* solver): Expansion(solver), initialized(false),
    symmetry(E_UNSPECIFIED), polarization(E_UNSPECIFIED) {}

size_t ExpansionPW2D::lcount() const {
    return SOLVER->getLayersPoints().size();
}

void ExpansionPW2D::init()
{
    auto geometry = SOLVER->getGeometry();

    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    left = geometry->getChild()->getBoundingBox().lower[0];
    right = geometry->getChild()->getBoundingBox().upper[0];

    size_t refine = SOLVER->refine, M;
    if (refine == 0) refine = 1;

    symmetric = separated = false;
    if (symmetry != E_UNSPECIFIED || polarization != E_UNSPECIFIED) {
        // Test for off-diagonal NR components in which case we cannot use neither symmetry nor separation
        bool off_diagonal = false;
        double dx =  (right-left) / (2 * SOLVER->getSize() * refine);
        for (const RectilinearAxis& axis1: SOLVER->getLayersPoints()) {
            for (double x = left+dx/2; x < right; x += dx) {
                Tensor3<dcomplex> nr = geometry->getMaterial(vec(x,axis1[0]))->NR(real(SOLVER->getWavelength()), 300.);
                if (nr.c01 != 0. || nr.c10 != 0.) { off_diagonal = true; break; }
            }
            if (off_diagonal) break;
        }
        if (symmetry != E_UNSPECIFIED) {
            if (!geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
                throw BadInput(solver->getId(), "Symmetry not allowed for asymmetric structure");
            if (off_diagonal)
                throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
            symmetric = true;
        }
        if (polarization != E_UNSPECIFIED) {
            if (off_diagonal)
                throw BadInput(solver->getId(), "Single polarization not allowed for structure with non-diagonal NR tensor");
            separated = true;
        }
    }

    if (geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN)) {
        if (right <= 0) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0) throw BadMesh(SOLVER->getId(), "Symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric) left = -right;
    }

    if (!periodic) {
        // Add PMLs
        if (!symmetric) left -= SOLVER->pml.size + SOLVER->pml.shift;
        right += SOLVER->pml.size + SOLVER->pml.shift;
    }

    double L;
                                                            // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric) {                                       //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        L = right - left;                                   //  ^ ^ ^ ^ ^
        N = 2 * SOLVER->getSize() + 1;                      // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nN = 4 * SOLVER->getSize() + 1;
        M = refine * nN;                                    // N = 3  nN = 5  refine = 4  M = 20
        double dx = 0.5 * L * (refine-1) / M;               // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        xmesh = RegularAxis(left-dx, right-dx-L/M, M);      //  ^ ^ ^ ^
        xpoints = RegularAxis(left, right-L/N, N);          // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    } else {
        L = 2 * right;
        N = SOLVER->getSize() + 1;
        nN = 2 * SOLVER->getSize() + 1;                     // N = 3  nN = 5  refine = 4  M = 20
        M = refine * nN;                                    // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        double dx = 0.25 * L / M;                           //  ^ ^ ^ ^
        xmesh = RegularAxis(left + dx, right - dx, M);      // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        dx = 0.25 * L / N;
        xpoints = RegularAxis(left + dx, right - dx, N);
    }

    SOLVER->writelog(LOG_DETAIL, "Creating%3%%4% expansion with %1% plane-waves (matrix size: %2%)",
                     N, matrixSize(), symmetric?" symmetric":"", separated?" separated":"");

    matFFT = FFT::Forward1D(5, nN, symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE);

    // Compute permeability coefficients
    mag.reset(nN, Tensor2<dcomplex>(0.));
    if (periodic) {
        mag[0].c00 = 1.; mag[0].c11 = 1.; // constant 1
    } else {
        DataVector<double> Sy(M, 0.);   // PML coeffs for mu
        // Add PMLs
        SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure width: %1%um)", L);
        double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
        if (symmetric) pil = 0;
        else pil = std::lower_bound(xmesh.begin(), xmesh.end(), pl) - xmesh.begin();
        pir = std::lower_bound(xmesh.begin(), xmesh.end(), pr) - xmesh.begin();
        for (size_t i = 0; i < pil; ++i) {
            double h = (pl - xmesh[i]) / SOLVER->pml.size;
            Sy[i] = SOLVER->pml.extinction * pow(h, SOLVER->pml.order);
        }
        for (size_t i = pir+1; i < xmesh.size(); ++i) {
            double h = (xmesh[i] - pr) / SOLVER->pml.size;
            Sy[i] = SOLVER->pml.extinction * pow(h, SOLVER->pml.order);
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
        FFT::Forward1D(2, nN, symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(mag.data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI / L; bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN; ++i) {
                int k = i; if (k > nN/2) k -= nN;
                mag[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = lcount();
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);
    #pragma omp parallel for
    for (size_t l = 0; l < nlayers; ++l)
        getMaterialCoefficients(l);

    initialized = true;
}

void ExpansionPW2D::free() {
    coeffs.clear();
    initialized = false;
}

void ExpansionPW2D::getMaterialCoefficients(size_t l)
{
    if (isnan(real(SOLVER->getWavelength())) || isnan(imag(SOLVER->getWavelength())))
        throw BadInput(SOLVER->getId(), "No wavelength set in SOLVER");

    auto geometry = SOLVER->getGeometry();
    const RectilinearAxis& axis1 = SOLVER->getLayerPoints(l);

    size_t refine = SOLVER->refine;
    size_t M = refine * nN;

    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer %1% (sampled at %2% points)", l, M);

    DataVector<Tensor3<dcomplex>> NR(M);

    RectilinearMesh2D mesh(xmesh, axis1, RectilinearMesh2D::ORDER_TRANSPOSED);
    auto temperature = SOLVER->inTemperature(mesh);
    double lambda = real(SOLVER->getWavelength());
    double maty = axis1[0]; // at each point along any vertical axis material is the same
    for (size_t i = 0; i < M; ++i) {
        auto material = geometry->getMaterial(Vec<2>(xmesh[i],maty));
        // assert([&]()->bool{for(auto y: axis1)if(geometry->getMaterial(Vec<2>(xmesh[i],y))!=material)return false; return true;}());
        double T = 0.; // average temperature in all vertical points
        for (size_t j = i * axis1.size(), end = (i+1) * axis1.size(); j != end; ++j) T += temperature[j];
        T /= axis1.size();
        #pragma omp critical
        NR[i] = material->NR(lambda, T);
    }

    for (Tensor3<dcomplex>& val: NR) val.sqr_inplace(); // make epsilon from NR

    // Add PMLs
    if (!periodic) {
        Tensor3<dcomplex> ref;
        double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
        ref = NR[pil];
        for (size_t i = 0; i < pil; ++i) {
            double h = (pl - xmesh[i]) / SOLVER->pml.size;
            dcomplex sy(1., SOLVER->pml.extinction * pow(h, SOLVER->pml.order));
            NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
        }
        ref = NR[min(pir,xmesh.size()-1)];
        for (size_t i = pir+1; i < xmesh.size(); ++i) {
            double h = (xmesh[i] - pr) / SOLVER->pml.size;
            dcomplex sy(1., SOLVER->pml.extinction * pow(h, SOLVER->pml.order));
            NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
        }
    }

    // Average material parameters
    coeffs[l].reset(nN, Tensor3<dcomplex>(0.));
    double factor = 1. / refine;
    for (size_t i = 0; i != nN; ++i) {
        for (size_t j = refine*i, end = refine*(i+1); j != end; ++j)
            coeffs[l][i] += Tensor3<dcomplex>(NR[j].c00, 1./NR[j].c11, NR[j].c22, NR[j].c01, NR[j].c10);
        coeffs[l][i] *= factor;
        coeffs[l][i].c11 = 1. / coeffs[l][i].c11; // We were averaging inverses of c11 (xx)
        coeffs[l][i].c22 = 1. / coeffs[l][i].c22; // We need inverse of c22 (yy)
    }

    // Check if the layer is uniform
    if (periodic) {
        diagonals[l] = true;
        for (size_t i = 1; i != nN; ++i) {
            Tensor3<dcomplex> diff = coeffs[l][i] - coeffs[l][0];
            if (!(is_zero(diff.c00) && is_zero(diff.c11) && is_zero(diff.c22) &&
                  is_zero(diff.c01) && is_zero(diff.c10))) {
                diagonals[l] = false;
                break;
            }
        }
    } else
        diagonals[l] = false;

    if (diagonals[l]) {
        solver->writelog(LOG_DETAIL, "Layer %1%  uniform", l);
        for (size_t i = 1; i != nN; ++i) coeffs[l][i] = Tensor3<dcomplex>(0.);
    } else {
        // Perform FFT
        matFFT.execute(reinterpret_cast<dcomplex*>(coeffs[l].data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI / ((right-left) * (symmetric? 2 : 1)); bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN; ++i) {
                int k = i; if (k > nN/2) k -= nN;
                coeffs[l][i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }
}


DataVector<const Tensor3<dcomplex>> ExpansionPW2D::getMaterialNR(size_t l, const RectilinearAxis mesh, InterpolationMethod interp)
{
    double L = right - left;
    DataVector<Tensor3<dcomplex>> result;
    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        result.reset(mesh.size(), Tensor3<dcomplex>(0.));
        if (!symmetric) {
            for (int k = -int(nN)/2, end = int(nN+1)/2; k != end; ++k) {
                size_t j = (k>=0)? k : k + nN;
                for (size_t i = 0; i != mesh.size(); ++i) {
                    result[i] += coeffs[l][j] * exp(2*M_PI * k * I * (mesh[i]-left) / L);
                }
            }
        } else {
            for (size_t i = 0; i != mesh.size(); ++i) {
                result[i] += coeffs[l][0];
                for (int k = 1; k != nN; ++k) {
                    result[i] += 2. * coeffs[l][k] * cos(M_PI * k * mesh[i] / L);
                }
            }
        }
    } else {
        DataVector<Tensor3<dcomplex>> params(symmetric? nN : nN+1);
        std::copy(coeffs[l].begin(), coeffs[l].end(), params.begin());
        FFT::Backward1D(5, nN, symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(params.data()));
        RegularAxis cmesh;
        if (symmetric) {
            double dx = 0.5 * (right-left) / nN;
            cmesh.reset(left + dx, right - dx, nN);
        } else {
            cmesh.reset(left, right, nN+1);
            auto old = coeffs[l];
            params[nN] = params[0];
        }
        RegularMesh2D src_mesh(cmesh, RegularAxis(0,0,1));
        RectilinearMesh2D dst_mesh(mesh, RectilinearAxis({0}));
        result = interpolate(src_mesh, params, WrappedMesh<2>(dst_mesh, SOLVER->getGeometry()), interp);
    }
    for (Tensor3<dcomplex>& eps: result) {
        eps.c22 = 1. / eps.c22;
        eps.sqrt_inplace();
    }
    return result;
}



void ExpansionPW2D::getMatrices(size_t l, dcomplex k0, dcomplex beta, dcomplex kx, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);

    int order = SOLVER->getSize();
    dcomplex f = 1. / k0, k02 = k0*k0;
    double b = 2*M_PI / (right-left) * (symmetric? 0.5 : 1.0);

    // Ez represents -Ez

    if (separated) {
        //TODO
        if (symmetric) {
            // Separated symmetric
            std::fill_n(RE.data(), N*N, dcomplex(0.));
            std::fill_n(RH.data(), N*N, dcomplex(0.));
            if (polarization == E_LONG) {                   // Ez & Hx
                for (int i = 0; i <= order; ++i) {
                    double gi = b * double(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = abs(i-j);   double gj = b * double(j);
                        dcomplex fz = (j < 0 && symmetry == E_TRAN)? -f : f;
                        int aj = abs(j);
                        RE(iH(i), iE(aj)) += fz * (- gi * gj * imuyy(l,ij) + k02 * epszz(l,ij) );
                        RH(iE(i), iH(aj)) += fz *                            k02 * muxx(l,ij);
                    }
                }
            } else {                                        // Ex & Hz
                for (int i = 0; i <= order; ++i) {
                    double gi = b * double(i);
                    for (int j = -order; j <= order; ++j) {
                        int ij = abs(i-j);   double gj = b * double(j);
                        dcomplex fx = (j < 0 && symmetry == E_LONG)? -f : f;
                        int aj = abs(j);
                        RE(iH(i), iE(aj)) += fx *                             k02 * epsxx(l,ij);
                        RH(iE(i), iH(aj)) += fx * (- gi * gj * iepsyy(l,ij) + k02 * muzz(l,ij) );
                    }
                }
            }
        } else {
            // Separated asymmetric
            if (polarization == E_LONG) {                   // Ez & Hx
                for (int i = -order; i <= order; ++i) {
                    dcomplex gi = b * double(i) - kx;
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
                        RE(iH(i), iE(j)) = f * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
                        RH(iE(i), iH(j)) = f *                              k02 * muxx(l,ij);
                    }
                }
            } else {                                        // Ex & Hz
                for (int i = -order; i <= order; ++i) {
                    dcomplex gi = b * double(i) - kx;
                    for (int j = -order; j <= order; ++j) {
                        int ij = i-j;   dcomplex gj = b * double(j) - kx;
                        RE(iH(i), iE(j)) = f *                               k02 * epsxx(l,ij);
                        RH(iE(i), iH(j)) = f * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
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
                double gi = b * double(i);
                for (int j = -order; j <= order; ++j) {
                    int ij = abs(i-j);   double gj = b * double(j);
                    dcomplex fx = (j < 0 && symmetry == E_LONG)? -f : f;
                    dcomplex fz = (j < 0 && symmetry == E_TRAN)? -f : f;
                    int aj = abs(j);
                    RE(iHz(i), iEx(aj)) += fx * (- beta*beta * imuyy(l,ij) + k02 * epsxx(l,ij) );
                    RE(iHx(i), iEx(aj)) += fx * (  beta* gi  * imuyy(l,ij)                     );
                    RE(iHz(i), iEz(aj)) += fz * (  beta* gj  * imuyy(l,ij)                     );
                    RE(iHx(i), iEz(aj)) += fz * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
                    RH(iEx(i), iHz(aj)) += fx * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
                    RH(iEz(i), iHz(aj)) += fx * (- beta* gj  * iepsyy(l,ij)                    );
                    RH(iEx(i), iHx(aj)) += fz * (- beta* gi  * iepsyy(l,ij)                    );
                    RH(iEz(i), iHx(aj)) += fz * (- beta*beta * iepsyy(l,ij) + k02 * muxx(l,ij) );
                    // if(RE(iHz(i), iEx(j)) == 0.) RE(iHz(i), iEx(j)) = 1e-32;
                    // if(RE(iHx(i), iEz(j)) == 0.) RE(iHx(i), iEz(j)) = 1e-32;
                    // if(RH(iEx(i), iHz(j)) == 0.) RH(iEx(i), iHz(j)) = 1e-32;
                    // if(RH(iEz(i), iHx(j)) == 0.) RH(iEz(i), iHx(j)) = 1e-32;
                }
            }
        } else {
            // Full asymmetric
            for (int i = -order; i <= order; ++i) {
                dcomplex gi = b * double(i) - kx;
                for (int j = -order; j <= order; ++j) {
                    int ij = i-j;   dcomplex gj = b * double(j) - kx;
                    RE(iHz(i), iEx(j)) = f * (- beta*beta * imuyy(l,ij) + k02 * epsxx(l,ij) );
                    RE(iHx(i), iEx(j)) = f * (  beta* gi  * imuyy(l,ij) - k02 * epszx(l,ij) );
                    RE(iHz(i), iEz(j)) = f * (  beta* gj  * imuyy(l,ij) - k02 * epsxz(l,ij) );
                    RE(iHx(i), iEz(j)) = f * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
                    RH(iEx(i), iHz(j)) = f * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
                    RH(iEz(i), iHz(j)) = f * (- beta* gj  * iepsyy(l,ij)                    );
                    RH(iEx(i), iHx(j)) = f * (- beta* gi  * iepsyy(l,ij)                    );
                    RH(iEz(i), iHx(j)) = f * (- beta*beta * iepsyy(l,ij) + k02 * muxx(l,ij) );
                    // if(RE(iHz(i), iEx(j)) == 0.) RE(iHz(i), iEx(j)) = 1e-32;
                    // if(RE(iHx(i), iEz(j)) == 0.) RE(iHx(i), iEz(j)) = 1e-32;
                    // if(RH(iEx(i), iHz(j)) == 0.) RH(iEx(i), iHz(j)) = 1e-32;
                    // if(RH(iEz(i), iHx(j)) == 0.) RH(iEz(i), iHx(j)) = 1e-32;
                }
            }
        }
    }
}


DataVector<Vec<3,dcomplex>> ExpansionPW2D::fieldE(size_t l, const Mesh& dst_mesh, dcomplex k0, dcomplex beta, dcomplex kx,
                                                 const cvector& E, const cvector& H, InterpolationMethod method)
{
    int order = SOLVER->getSize();
    double b = 2*M_PI / (right-left) * (symmetric? 0.5 : 1.0);
    DataVector<Vec<3,dcomplex>> src(N+1);
    assert(dynamic_cast<const LevelMeshAdapter<2>*>(&dst_mesh));
    double vpos = static_cast<const LevelMeshAdapter<2>&>(dst_mesh).vpos();
    
    if (separated) {
    } else {
        if (symmetric) {
            for (int i = 0; i <= order; ++i) {
                src[iE(i)].lon() = E[iEz(i)];
                src[iE(i)].tran() = E[iEx(i)];
                src[iE(i)].vert() = 0.;
//                 TODO
//                 for (int j = -order; j <= order; ++j) {
//                     int aj = abs(j);
//                     dcomplex gj = b * double(j);
//                     src[iE(i)].vert() = - iepsyy(l, abs(i-j)) * (gj*H[iHz[aj]] + beta*H[iHx[aj]]);
//                 }
            }
        } else {
            for (int i = -order; i <= order; ++i) {
                src[iE(i)].lon() = E[iEz(i)];
                src[iE(i)].tran() = E[iEx(i)];
                src[iE(i)].vert() = - iepsyy(l, i) * beta * H[iHx(i)];
                for (int j = -order; j <= order; ++j) {
                    dcomplex gj = b * double(j) - kx;
                    src[iE(i)].vert() -= iepsyy(l, i-j) * gj * H[iHz(j)];
                }
            }
        }
    }
    
    if (symmetric) {
        return DataVector<Vec<3,dcomplex>>();
    } else {
        FFT::Backward1D fft(3, N, FFT::SYMMETRY_NONE);
        fft.execute(reinterpret_cast<dcomplex*>(src.data()));
        src[N] = src[0];
        RegularMesh2D src_mesh(RegularAxis(vpos, vpos, 1), RegularAxis(left, right, N+1));
        return interpolate(src_mesh, src, WrappedMesh<2>(static_cast<const MeshD<2>&>(dst_mesh), SOLVER->getGeometry()),
                           defInterpolation<INTERPOLATION_SPLINE>(method), false);
    }
}


DataVector<Vec<3,dcomplex>> ExpansionPW2D::fieldH(size_t l, const Mesh& dst_mesh, dcomplex k0, dcomplex klong, dcomplex ktran,
                                                 const cvector& E, const cvector& H, InterpolationMethod method)
{
//     return DataVector<Vec<3,dcomplex>>(dst_mesh.size(), Vec<3,dcomplex>(0.,0.,0.));
    int order = SOLVER->getSize();
    double b = 2*M_PI / (right-left) * (symmetric? 0.5 : 1.0);
    DataVector<Vec<3,dcomplex>> src(N+1);
    assert(dynamic_cast<const LevelMeshAdapter<2>*>(&dst_mesh));
    double vpos = static_cast<const LevelMeshAdapter<2>&>(dst_mesh).vpos();
    
    if (separated) {
    } else {
        if (symmetric) {
        } else {
            for (int i = -order; i <= order; ++i) {
                src[iH(i)].lon() = H[iHz(i)];
                src[iH(i)].tran() = H[iHx(i)];
                src[iH(i)].vert() = 0.;
            }
        }
    }
    
    if (symmetric) {
        return DataVector<Vec<3,dcomplex>>();
    } else {
        FFT::Backward1D fft(3, N, FFT::SYMMETRY_NONE);
        fft.execute(reinterpret_cast<dcomplex*>(src.data()));
        src[N] = src[0];
        RegularMesh2D src_mesh(RegularAxis(vpos, vpos, 1), RegularAxis(left, right, N+1));
        return interpolate(src_mesh, src, WrappedMesh<2>(static_cast<const MeshD<2>&>(dst_mesh), SOLVER->getGeometry()),
                           defInterpolation<INTERPOLATION_SPLINE>(method), false);
    }
}


}}} // namespace plask
