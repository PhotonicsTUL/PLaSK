#include "expansion_pw3d.h"
#include "fourier_reflection_3d.h"
#include "mesh_adapter.h"

#define SOLVER static_cast<FourierReflection3D*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionPW3D::ExpansionPW3D(FourierReflection3D* solver): Expansion(solver), initialized(false),
    symmetry_long(E_UNSPECIFIED), symmetry_tran(E_UNSPECIFIED) {}

size_t ExpansionPW3D::lcount() const {
    return SOLVER->getLayersPoints().size();
}

void ExpansionPW3D::init()
{
    auto geometry = SOLVER->getGeometry();

    periodic_long = geometry->isPeriodic(Geometry3D::DIRECTION_LONG);
    periodic_tran = geometry->isPeriodic(Geometry3D::DIRECTION_TRAN);

    left = geometry->getChild()->getBoundingBox().lower[0];
    right = geometry->getChild()->getBoundingBox().upper[0];
    back = geometry->getChild()->getBoundingBox().lower[1];
    front = geometry->getChild()->getBoundingBox().upper[1];

    size_t refl = SOLVER->refine_long, reft = SOLVER->refine_tran, Ml, Mt;
    if (refl == 0) refl = 1;  if (reft == 0) reft = 1;

    symmetric_long = symmetric_tran = false;
    if (symmetry_long != E_UNSPECIFIED) {
        if (!geometry->isSymmetric(Geometry3D::DIRECTION_LONG))
            throw BadInput(solver->getId(), "Longitudinal symmetry not allowed for asymmetric structure");
        symmetric_long = true;
    }
    if (symmetry_tran != E_UNSPECIFIED) {
        if (!geometry->isSymmetric(Geometry3D::DIRECTION_TRAN))
            throw BadInput(solver->getId(), "Transverse symmetry not allowed for asymmetric structure");
        symmetric_tran = true;
    }

    if (geometry->isSymmetric(Geometry3D::DIRECTION_LONG)) {
        if (front <= 0) {
            back = -back; front = -front;
            std::swap(back, front);
        }
        if (back != 0) throw BadMesh(SOLVER->getId(), "Longitudinally symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_long) back = -front;
    }
    if (geometry->isSymmetric(Geometry3D::DIRECTION_TRAN)) {
        if (right <= 0) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0) throw BadMesh(SOLVER->getId(), "Transversely symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_tran) left = -right;
    }

    if (!periodic_long) {
        // Add PMLs
        if (!symmetric_long) back -= SOLVER->pml_long.size + SOLVER->pml_long.shift;
        front += SOLVER->pml_long.size + SOLVER->pml_long.shift;
    }
    if (!periodic_tran) {
        // Add PMLs
        if (!symmetric_tran) left -= SOLVER->pml_tran.size + SOLVER->pml_tran.shift;
        right += SOLVER->pml_tran.size + SOLVER->pml_tran.shift;
    }

    double Ll, Lt;

    if (!symmetric_long) {
        Ll = front - back;
        Nl = 2 * SOLVER->getLongSize() + 1;
        nNl = 4 * SOLVER->getLongSize() + 1;
        Ml = refl * nNl;
        double dx = 0.5 * Ll * (refl-1) / Ml;
        long_mesh = RegularAxis(back-dx, front-dx-Ll/Ml, Ml);
    } else {
        Ll = 2 * front;
        Nl = SOLVER->getLongSize() + 1;
        nNl = 2 * SOLVER->getLongSize() + 1;
        Ml = refl * nNl;
        double dx = 0.25 * Ll / Ml;
        long_mesh = RegularAxis(back + dx, front - dx, Ml);
    }                                                           // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric_tran) {                                      //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        Lt = right - left;                                      //  ^ ^ ^ ^ ^
        Nt = 2 * SOLVER->getTranSize() + 1;                     // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nNt = 4 * SOLVER->getTranSize() + 1;
        Mt = reft * nNt;                                        // N = 3  nN = 5  refine = 4  M = 20
        double dx = 0.5 * Lt * (reft-1) / Mt;                   // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        tran_mesh = RegularAxis(left-dx, right-dx-Lt/Mt, Mt);   //  ^ ^ ^ ^
    } else {                                                    // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        Lt = 2 * right;
        Nt = SOLVER->getTranSize() + 1;
        nNt = 2 * SOLVER->getTranSize() + 1;                    // N = 3  nN = 5  refine = 4  M = 20
        Mt = reft * nNt;                                        // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        double dx = 0.25 * Lt / Mt;                             //  ^ ^ ^ ^
        tran_mesh = RegularAxis(left + dx, right - dx, Mt);     // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    }

    SOLVER->writelog(LOG_DETAIL, "Creating expansion%4% with %1%x%2% plane-waves (matrix size: %3%)", Nl, Nt, matrixSize(),
                     (!symmetric_long && !symmetric_tran)? "" :
                     (symmetric_long && symmetric_tran)? " symmetric in longitudinal and transverse directions" :
                     (!symmetric_long && symmetric_tran)? " symmetric in transverse direction" : " symmetric in longitudinal direction"
                    );

    matFFT = FFT::Forward2D(5, nNl, nNt, symmetric_long? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE, symmetric_tran? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE);

//     // Compute permeability coefficients
//     mag.reset(nN, Tensor2<dcomplex>(0.));
//     if (periodic) {
//         mag[0].c00 = 1.; mag[0].c11 = 1.; // constant 1
//     } else {
//         DataVector<dcomplex> Sy(M, 1.);   // PML coeffs for mu
//         // Add PMLs
//         SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure width: %1%um)", L);
//         double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
//         if (symmetric) pil = 0;
//         else pil = std::lower_bound(xmesh.begin(), xmesh.end(), pl) - xmesh.begin();
//         pir = std::lower_bound(xmesh.begin(), xmesh.end(), pr) - xmesh.begin();
//         for (size_t i = 0; i < pil; ++i) {
//             double h = (pl - xmesh[i]) / SOLVER->pml.size;
//             Sy[i] = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
//         }
//         for (size_t i = pir+1; i < xmesh.size(); ++i) {
//             double h = (xmesh[i] - pr) / SOLVER->pml.size;
//             Sy[i] = 1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order);
//         }
//         // Average mu
//         std::fill(mag.begin(), mag.end(), Tensor2<dcomplex>(0.));
//         for (size_t i = 0; i != nN; ++i) {
//             for (size_t j = refine*i, end = refine*(i+1); j != end; ++j) {
//                 mag[i] += Tensor2<dcomplex>(Sy[j], 1./Sy[j]);
//             }
//             mag[i] /= refine;
//         }
//         // Compute FFT
//         FFT::Forward1D(2, nN, symmetric? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(mag.data()));
//         // Smooth coefficients
//         if (SOLVER->smooth) {
//             double bb4 = M_PI / L; bb4 *= bb4;   // (2π/L)² / 4
//             for (size_t i = 0; i != nN; ++i) {
//                 int k = i; if (k > nN/2) k -= nN;
//                 mag[i] *= exp(-SOLVER->smooth * bb4 * k * k);
//             }
//         }
//     }
//
    // Allocate memory for expansion coefficients
    size_t nlayers = lcount();
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);

    initialized = true;
}

void ExpansionPW3D::free() {
    coeffs.clear();
    initialized = false;
}

void ExpansionPW3D::layerMaterialCoefficients(size_t l)
{
    if (isnan(real(SOLVER->getWavelength())) || isnan(imag(SOLVER->getWavelength())))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();
    const RectilinearAxis& axis2 = SOLVER->getLayerPoints(l);

    const double Lt = right - left, Ll = front - back;
    const size_t refl = (SOLVER->refine_long)? SOLVER->refine_long : 1,
                 reft = (SOLVER->refine_tran)? SOLVER->refine_tran : 1;
    const size_t Ml = refl * nNl,  Mt = reft * nNt;
    const size_t M = Ml * Mt;

    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer %1% (sampled at %2%x%3% points)", l, Ml, Mt);

    DataVector<Tensor3<dcomplex>> NR(M);

    RectangularMesh<3> mesh(make_shared<RegularAxis>(long_mesh), make_shared<RegularAxis>(tran_mesh), make_shared<RectilinearAxis>(axis2), RectangularMesh<3>::ORDER_012);

    double lambda = real(SOLVER->getWavelength());

    auto temperature = SOLVER->inTemperature(mesh);

    DataVector<const double> gain;
    bool have_gain = false;
    if (SOLVER->inGain.hasProvider()) {
        gain = SOLVER->inGain(mesh, lambda);
        have_gain = true;
    }

    double matv = axis2[0]; // at each point along any vertical axis material is the same
    for (size_t it = 0, i = 0; it < Mt; ++it) {
        for (size_t il = 0; il < Ml; ++il, ++i) {
            auto material = geometry->getMaterial(vec(long_mesh[il], tran_mesh[it], matv));
            double T = 0.; // average temperature in all vertical points
            for (size_t j = mesh.index(il, it, 0), end = mesh.index(il, it, axis2.size()); j != end; ++j) T += temperature[j];
            T /= axis2.size();
            #pragma omp critical
            NR[i] = material->NR(lambda, T);
            if (NR[i].c10 != 0. || NR[i].c01 != 0.) {
                if (symmetric_long || symmetric_tran) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
            }
            if (have_gain) {
                auto roles = geometry->getRolesAt(vec(long_mesh[il], tran_mesh[it], matv));
                if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                    double g = 0.; // average gain in all vertical points
                    for (size_t j = mesh.index(il, it, 0) * axis2.size(), end = mesh.index(il, it, axis2.size())+1; j != end; ++j) g += gain[j];
                    double ni = lambda * g/axis2.size() * 7.95774715459e-09;
                    NR[i].c00.imag(ni); NR[i].c11.imag(ni); NR[i].c22.imag(ni); NR[i].c01.imag(0.); NR[i].c10.imag(0.);
                }
            }
        }
    }

    for (Tensor3<dcomplex>& val: NR) val.sqr_inplace(); // make epsilon from NR

    size_t nN = nNl * nNt;

//     // Add PMLs
//     if (!periodic) {
//         Tensor3<dcomplex> ref;
//         double pl = left + SOLVER->pml.size, pr = right - SOLVER->pml.size;
//         ref = NR[pil];
//         for (size_t i = 0; i < pil; ++i) {
//             double h = (pl - xmesh[i]) / SOLVER->pml.size;
//             dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
//             NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
//         }
//         ref = NR[min(pir,xmesh.size()-1)];
//         for (size_t i = pir+1; i < xmesh.size(); ++i) {
//             double h = (xmesh[i] - pr) / SOLVER->pml.size;
//             dcomplex sy(1. + (SOLVER->pml.factor-1.)*pow(h, SOLVER->pml.order));
//             NR[i] = Tensor3<dcomplex>(ref.c00*sy, ref.c11/sy, ref.c22*sy);
//         }
//     }

    // Average material parameters
    coeffs[l].reset(nN, Tensor3<dcomplex>(0.));

    const double normlim = min(Ll/nNl, Lt/nNt) * 1e-9;

    for (size_t it = 0; it != nNt; ++it) {

        size_t tbegin = reft * it; size_t tend = tbegin + reft;
        double tran0 = 0.5 * (tran_mesh[tbegin] + tran_mesh[tend-1]);

        for (size_t il = 0; il != nNl; ++il) {
            size_t lbegin = refl * il; size_t lend = lbegin + refl;
            double long0 = 0.5 * (long_mesh[lbegin] + long_mesh[lend-1]);

            // Compute surface normal
            Vec<2> norm(0.,0.);
            for (size_t t = tbegin; t != tend; ++t) {
                size_t toff = Ml * t;
                for (size_t l = lbegin; l != lend; ++l) {
                    const auto& n = NR[toff + l];
                    norm += (real(n.c00) + real(n.c11)) * vec(long_mesh[l] - long0, tran_mesh[t] - tran0);
                }
            }
            double a = abs(norm);
            if (a < normlim) {
                // Nothing to average
                //coeffs[l][nNl * it + il] = NR[Ml * (tbegin+tend)/2 + (lbegin+lend)/2];
                coeffs[l][nNl * it + il] = Tensor3<dcomplex>(0.); //TODO just for testing
            } else {
                norm /= a;
                auto& nr = coeffs[l][nNl * it + il];

                // Average permittivity tensor according to:
                // [ S. G. Johnson and J. D. Joannopoulos, Opt. Express, vol. 8, pp. 173-190 (2001) ]
                nr = Tensor3<dcomplex>(norm.c0, norm.c1, 0.); //TODO just for testing
            }
        }
    }

    // Check if the layer is uniform
    if (periodic_tran && periodic_long) {
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
        solver->writelog(LOG_DETAIL, "Layer %1% is uniform", l);
        for (size_t i = 1; i != nN; ++i) coeffs[l][i] = Tensor3<dcomplex>(0.);
    } else {
        // Perform FFT
        matFFT.execute(reinterpret_cast<dcomplex*>(coeffs[l].data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = 2.*M_PI / ((right-left) * (symmetric_tran? 2 : 1)) / ((front-back) * (symmetric_long? 2 : 1));
            bb4 *= bb4; bb4 *= 0.25;  // (2π/Lt)² (2π/Ll)² / 4
            for (size_t it = 0; it != nNl; ++it) {
                int kt = it; if (kt > nNt/2) kt -= nNt;
                for (size_t il = 0; il != nNl; ++il) {
                    int kl = il; if (kl > nNl/2) kl -= nNl;
                    coeffs[l][nNl*it+il] *= exp(-SOLVER->smooth * bb4 * kt*kt * kl*kl);
                }
            }
        }
    }
}


DataVector<const Tensor3<dcomplex>> ExpansionPW3D::getMaterialNR(size_t lay, RectilinearAxis lmesh, RectilinearAxis tmesh, InterpolationMethod interp)
{
    DataVector<Tensor3<dcomplex>> result;
//     if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
//         const double Lt = right - left, Ll = front - back;
//         if (!symmetric) {
//             result.reset(mesh.size(), Tensor3<dcomplex>(0.));
//             for (int k = -int(nN)/2, end = int(nN+1)/2; k != end; ++k) {
//                 size_t j = (k>=0)? k : k + nN;
//                 for (size_t i = 0; i != mesh.size(); ++i) {
//                     result[i] += coeffs[lay][j] * exp(2*M_PI * k * I * (mesh[i]-left) / L);
//                 }
//             }
//         } else {
//             result.reset(mesh.size());
//             for (size_t i = 0; i != mesh.size(); ++i) {
//                 result[i] = coeffs[lay][0];
//                 for (int k = 1; k != nN; ++k) {
//                     result[i] += 2. * coeffs[lay][k] * cos(M_PI * k * mesh[i] / L);
//                 }
//             }
//         }
//     } else {
        size_t nl = symmetric_long? nNl : nNl+1, nt = symmetric_tran? nNt : nNt+1;
        DataVector<Tensor3<dcomplex>> params(nl * nt);
        for (size_t t = 0; t != nNt; ++t) {
            size_t op = nl * t, oc = nNl * t;
            for (size_t l = 0; l != nNl; ++l) {
                params[op+l] = coeffs[lay][oc+l];
            }
        }
        FFT::Backward2D(5, nNl, nNt,
                        symmetric_long? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE,
                        symmetric_tran? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_NONE,
                        0, nl
                       )
            .execute(reinterpret_cast<dcomplex*>(params.data()));
        shared_ptr<RegularAxis> lcmesh = make_shared<RegularAxis>(), tcmesh = make_shared<RegularAxis>();
        if (symmetric_long) {
            double dx = 0.5 * (front-back) / nNl;
            lcmesh->reset(back+dx, front-dx, nNl);
        } else {
            lcmesh->reset(back, front, nNl+1);
            for (size_t l = 0, last = nl*nNt; l != nNl; ++l) params[last+l] = params[l];
        }
        if (symmetric_tran) {
            double dx = 0.5 * (right-left) / nNt;
            tcmesh->reset(left+dx, right-dx, nNt);
        } else {
            tcmesh->reset(left, right, nNt+1);
            for (size_t t = 0, end = nl*nt; t != end; t += nl) params[nNl+t] = params[t];
        }
        RectangularMesh<3> src_mesh(lcmesh, tcmesh, make_shared<RegularAxis>(0,0,1));
        RectangularMesh<3> dst_mesh(make_shared<RectilinearAxis>(std::move(lmesh)), make_shared<RectilinearAxis>(std::move(tmesh)), shared_ptr<RectilinearAxis>(new RectilinearAxis{0}));
        const bool ignore_symmetry[3] = { !symmetric_long, !symmetric_tran, false };
        result = interpolate(src_mesh, params, WrappedMesh<3>(dst_mesh, SOLVER->getGeometry(), ignore_symmetry), interp);
//     }
//     for (Tensor3<dcomplex>& eps: result) {
//         eps.c22 = 1. / eps.c22;
//         eps.sqrt_inplace();
//     }
    return result;
}



void ExpansionPW3D::getMatrices(size_t l, dcomplex k0, dcomplex klong, dcomplex ktran, cmatrix& RE, cmatrix& RH)
{
//     assert(initialized);
//
//     int order = SOLVER->getSize();
//     dcomplex f = 1. / k0, k02 = k0*k0;
//     double b = 2*M_PI / (right-left) * (symmetric? 0.5 : 1.0);
//
//     // Ez represents -Ez
//
//     if (separated) {
//         if (symmetric) {
//             // Separated symmetric
//             std::fill_n(RE.data(), N*N, dcomplex(0.));
//             std::fill_n(RH.data(), N*N, dcomplex(0.));
//             if (polarization == E_LONG) {                   // Ez & Hx
//                 for (int i = 0; i <= order; ++i) {
//                     double gi = b * double(i);
//                     for (int j = -order; j <= order; ++j) {
//                         int ij = abs(i-j);   double gj = b * double(j);
//                         dcomplex fz = (j < 0 && symmetry == E_TRAN)? -f : f;
//                         int aj = abs(j);
//                         RE(iH(i), iE(aj)) += fz * (- gi * gj * imuyy(l,ij) + k02 * epszz(l,ij) );
//                         RH(iE(i), iH(aj)) += fz *                            k02 * muxx(l,ij);
//                     }
//                 }
//             } else {                                        // Ex & Hz
//                 for (int i = 0; i <= order; ++i) {
//                     double gi = b * double(i);
//                     for (int j = -order; j <= order; ++j) {
//                         int ij = abs(i-j);   double gj = b * double(j);
//                         dcomplex fx = (j < 0 && symmetry == E_LONG)? -f : f;
//                         int aj = abs(j);
//                         RE(iH(i), iE(aj)) += fx *                             k02 * epsxx(l,ij);
//                         RH(iE(i), iH(aj)) += fx * (- gi * gj * iepsyy(l,ij) + k02 * muzz(l,ij) );
//                     }
//                 }
//             }
//         } else {
//             // Separated asymmetric
//             if (polarization == E_LONG) {                   // Ez & Hx
//                 for (int i = -order; i <= order; ++i) {
//                     dcomplex gi = b * double(i) - kx;
//                     for (int j = -order; j <= order; ++j) {
//                         int ij = i-j;   dcomplex gj = b * double(j) - kx;
//                         RE(iH(i), iE(j)) = f * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
//                         RH(iE(i), iH(j)) = f *                              k02 * muxx(l,ij);
//                     }
//                 }
//             } else {                                        // Ex & Hz
//                 for (int i = -order; i <= order; ++i) {
//                     dcomplex gi = b * double(i) - kx;
//                     for (int j = -order; j <= order; ++j) {
//                         int ij = i-j;   dcomplex gj = b * double(j) - kx;
//                         RE(iH(i), iE(j)) = f *                               k02 * epsxx(l,ij);
//                         RH(iE(i), iH(j)) = f * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
//                     }
//                 }
//             }
//         }
//     } else {
//         if (symmetric) {
//             // Full symmetric
//             std::fill_n(RE.data(), 4*N*N, dcomplex(0.));
//             std::fill_n(RH.data(), 4*N*N, dcomplex(0.));
//             for (int i = 0; i <= order; ++i) {
//                 double gi = b * double(i);
//                 for (int j = -order; j <= order; ++j) {
//                     int ij = abs(i-j);   double gj = b * double(j);
//                     dcomplex fx = (j < 0 && symmetry == E_LONG)? -f : f;
//                     dcomplex fz = (j < 0 && symmetry == E_TRAN)? -f : f;
//                     int aj = abs(j);
//                     RE(iHz(i), iEx(aj)) += fx * (- klong*klong * imuyy(l,ij) + k02 * epsxx(l,ij) );
//                     RE(iHx(i), iEx(aj)) += fx * (  klong* gi  * imuyy(l,ij)                     );
//                     RE(iHz(i), iEz(aj)) += fz * (  klong* gj  * imuyy(l,ij)                     );
//                     RE(iHx(i), iEz(aj)) += fz * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
//                     RH(iEx(i), iHz(aj)) += fx * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
//                     RH(iEz(i), iHz(aj)) += fx * (- klong* gj  * iepsyy(l,ij)                    );
//                     RH(iEx(i), iHx(aj)) += fz * (- klong* gi  * iepsyy(l,ij)                    );
//                     RH(iEz(i), iHx(aj)) += fz * (- klong*klong * iepsyy(l,ij) + k02 * muxx(l,ij) );
//                     // if(RE(iHz(i), iEx(j)) == 0.) RE(iHz(i), iEx(j)) = 1e-32;
//                     // if(RE(iHx(i), iEz(j)) == 0.) RE(iHx(i), iEz(j)) = 1e-32;
//                     // if(RH(iEx(i), iHz(j)) == 0.) RH(iEx(i), iHz(j)) = 1e-32;
//                     // if(RH(iEz(i), iHx(j)) == 0.) RH(iEz(i), iHx(j)) = 1e-32;
//                 }
//             }
//         } else {
//             // Full asymmetric
//             for (int i = -order; i <= order; ++i) {
//                 dcomplex gi = b * double(i) - kx;
//                 for (int j = -order; j <= order; ++j) {
//                     int ij = i-j;   dcomplex gj = b * double(j) - kx;
//                     RE(iHz(i), iEx(j)) = f * (- klong*klong * imuyy(l,ij) + k02 * epsxx(l,ij) );
//                     RE(iHx(i), iEx(j)) = f * (  klong* gi  * imuyy(l,ij) - k02 * epszx(l,ij) );
//                     RE(iHz(i), iEz(j)) = f * (  klong* gj  * imuyy(l,ij) - k02 * epsxz(l,ij) );
//                     RE(iHx(i), iEz(j)) = f * (-  gi * gj  * imuyy(l,ij) + k02 * epszz(l,ij) );
//                     RH(iEx(i), iHz(j)) = f * (-  gi * gj  * iepsyy(l,ij) + k02 * muzz(l,ij) );
//                     RH(iEz(i), iHz(j)) = f * (- klong* gj  * iepsyy(l,ij)                    );
//                     RH(iEx(i), iHx(j)) = f * (- klong* gi  * iepsyy(l,ij)                    );
//                     RH(iEz(i), iHx(j)) = f * (- klong*klong * iepsyy(l,ij) + k02 * muxx(l,ij) );
//                     // if(RE(iHz(i), iEx(j)) == 0.) RE(iHz(i), iEx(j)) = 1e-32;
//                     // if(RE(iHx(i), iEz(j)) == 0.) RE(iHx(i), iEz(j)) = 1e-32;
//                     // if(RH(iEx(i), iHz(j)) == 0.) RH(iEx(i), iHz(j)) = 1e-32;
//                     // if(RH(iEz(i), iHx(j)) == 0.) RH(iEz(i), iHx(j)) = 1e-32;
//                 }
//             }
//         }
//     }
}
//
//
// void ExpansionPW3D::prepareField()
// {
//     field.reset(N + (symmetric? 0 : 1));
//     Component sym = (field_params.which == FieldParams::E)? symmetry : Component(2-symmetry);
//     fft_x = FFT::Backward1D(1, N, (sym==E_TRAN)? FFT::SYMMETRY_EVEN : FFT::SYMMETRY_ODD, 3);
//     fft_yz = FFT::Backward1D(1, N, (sym==E_TRAN)? FFT::SYMMETRY_ODD : FFT::SYMMETRY_EVEN, 3);
// }
//
// void ExpansionPW3D::cleanupField()
// {
//     field.reset();
//     fft_x = FFT::Backward1D();
//     fft_yz = FFT::Backward1D();
// }
//
// // TODO fields must be carefully verified

DataVector<Vec<3,dcomplex>> ExpansionPW3D::getField(size_t l, const Mesh& dst_mesh, const cvector& E, const cvector& H)
{
//     Component sym = (field_params.which == FieldParams::E)? symmetry : Component(2-symmetry);
//
//     const dcomplex klong = field_params.klong;
//     const dcomplex kx = field_params.ktran;
//
//     int order = SOLVER->getSize();
//     double b = 2*M_PI / (right-left) * (symmetric? 0.5 : 1.0);
//     assert(dynamic_cast<const LevelMeshAdapter<2>*>(&dst_mesh));
//     const MeshD<2>& dest_mesh = static_cast<const MeshD<2>&>(dst_mesh);
//     double vpos = static_cast<const LevelMeshAdapter<2>&>(dst_mesh).vpos();
//
//     int dt = (symmetric && field_params.method != INTERPOLATION_FOURIER && sym != E_TRAN)? 1 : 0;
//     int dl = (symmetric && field_params.method != INTERPOLATION_FOURIER && sym != E_LONG)? 1 : 0;
//
//     if (field_params.which == FieldParams::E) {
//         if (separated) {
//             if (polarization == E_TRAN) {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iE(i)].lon() = field[iE(i)].vert() = 0.;
//                     if (iE(i) != 0 || !dt) field[iE(i)-dt].tran() = E[iE(i)];
//                 }
//             } else {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iE(i)].tran() = 0.;
//                     if (iE(i) != 0 || !dl) {
//                         field[iE(i)-dl].lon() = - E[iE(i)];
//                         field[iE(i)-dl].vert() = - iepsyy(l, i) * klong * H[iH(i)] / field_params.k0;
//                     }
//                 }
//             }
//         } else {
//             for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                 if (iE(i) != 0 || !dt) field[iE(i)-dt].tran() = E[iEx(i)];
//                 if (iE(i) != 0 || !dl) {
//                     field[iE(i)-dl].lon() = - E[iEz(i)];
//                     field[iE(i)-dl].vert() = - iepsyy(l, i) * klong * H[iHx(i)];
//                     if (symmetric) {
//                         if (sym == E_LONG) {
//                             for (int j = -order; j <= order; ++j)
//                                 field[iE(i)-dl].vert() += iepsyy(l, abs(i-j)) * b*double(j) * H[iHz(abs(j))];
//                         } else {
//                             for (int j = 0; j <= order; ++j)
//                                 field[iE(i)-dl].vert() += iepsyy(l, abs(i-j)) * b*double(j) * H[iHz(j)];
//                             for (int j = -order; j < 0; ++j)
//                                 field[iE(i)-dl].vert() -= iepsyy(l, abs(i-j)) * b*double(j) * H[iHz(-j)];
//                         }
//                     } else {
//                         for (int j = -order; j <= order; ++j)
//                             field[iE(i)-dl].vert() += iepsyy(l, i-j) * (b*double(j)-kx) * H[iHz(j)];
//                     }
//                     field[iE(i)-dl].vert() /= field_params.k0;
//                 }
//             }
//         }
//     } else { // field_params.which == FieldParams::H
//         if (separated) {
//             if (polarization == E_LONG) {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iH(i)].lon() = field[iH(i)].vert() = 0.;
//                     if (iH(i) != 0 || !dt) field[iH(i)- dt].tran() = E[iH(i)];
//                 }
//             } else {
//                 for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                     field[iH(i)].tran() = 0.;
//                     if (iH(i) != 0 || !dl) {
//                         field[iH(i)- dl].lon() = E[iH(i)];
//                         field[iH(i)- dl].vert() = - imuyy(l, i) * klong * E[iE(i)] / field_params.k0;
//                     }
//                 }
//             }
//         } else {
//             for (int i = symmetric? 0 : -order; i <= order; ++i) {
//                 if (iH(i) != 0 || !dt) field[iH(i)- dt].tran() = H[iHx(i)];
//                 if (iH(i) != 0 || !dl) {
//                     field[iH(i)- dl].lon() = H[iHz(i)];
//                     field[iH(i)- dl].vert() = - imuyy(l, i) * klong * H[iEx(i)];
//                     if (symmetric) {
//                         if (sym == E_LONG) {
//                             for (int j = -order; j <= order; ++j)
//                                 field[iE(i)- dl].vert() += imuyy(l, abs(i-j)) * b*double(j) * H[iEz(abs(j))];
//                         } else {
//                             for (int j = 0; j <= order; ++j)
//                                 field[iE(i)- dl].vert() += imuyy(l, abs(i-j)) * b*double(j) * H[iEz(j)];
//                             for (int j = -order; j < 0; ++j)
//                                 field[iE(i)- dl].vert() -= imuyy(l, abs(i-j)) * b*double(j) * H[iEz(-j)];
//                         }
//                     } else {
//                         for (int j = -order; j <= order; ++j)
//                             field[iE(i)- dl].vert() += imuyy(l, i-j) * (b*double(j)-kx) * H[iEz(j)];
//                     }
//                     field[iH(i)].vert() /= field_params.k0;
//                 }
//             }
//         }
//     }
//
//     if (dt) { field[field.size()-1].tran() = 0.; }
//     if (dl) { field[field.size()-1].lon() = 0.; field[field.size()-1].vert() = 0.; }
//
//     if (field_params.method == INTERPOLATION_FOURIER) {
//         DataVector<Vec<3,dcomplex>> result(dest_mesh.size());
//         double L = right - left;
//         if (!symmetric) {
//             result.reset(dest_mesh.size(), Vec<3,dcomplex>(0.,0.,0.));
//             for (int k = -order; k <= order; ++k) {
//                 size_t j = (k>=0)? k : k + N;
//                 for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                     result[i] += field[j] * exp(2*M_PI * k * I * (dest_mesh[i][0]-left) / L);
//                 }
//             }
//         } else {
//             result.reset(dest_mesh.size());
//             for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                 result[i] = field[0];
//                 for (int k = 1; k <= order; ++k) {
//                     double cs =  2. * cos(M_PI * k * dest_mesh[i][0] / L);
//                     double sn =  2. * sin(M_PI * k * dest_mesh[i][0] / L);
//                     if (sym == E_TRAN) {
//                         result[i].lon() += field[k].lon() * sn;
//                         result[i].tran() += field[k].tran() * cs;
//                         result[i].vert() += field[k].vert() * sn;
//                     } else {
//                         result[i].lon() += field[k].lon() * cs;
//                         result[i].tran() += field[k].tran() * sn;
//                         result[i].vert() += field[k].vert() * cs;
//                     }
//                 }
//             }
//         }
//         return result;
//     } else {
//         if (symmetric) {
//             fft_x.execute(&(field.data()->tran()));
//             fft_yz.execute(&(field.data()->lon()));
//             fft_yz.execute(&(field.data()->vert()));
//             double dx = 0.5 * (right-left) / N;
//             RegularMesh3D src_mesh(RegularAxis(left+dx, right-dx, field.size()), RegularAxis(vpos, vpos, 1));
//             auto result = interpolate(src_mesh, field, WrappedMesh<2>(dest_mesh, SOLVER->getGeometry()),
//                                       defInterpolation<INTERPOLATION_SPLINE>(field_params.method), false);
//             double L = 2. * right;
//             if (sym == E_TRAN)
//                 for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                     double x = std::fmod(dest_mesh[i][0], L);
//                     if ((-right <= x && x < 0) || x > right) { result[i].lon() = -result[i].lon(); result[i].vert() = -result[i].vert(); }
//                 }
//             else
//                 for (size_t i = 0; i != dest_mesh.size(); ++i) {
//                     double x = std::fmod(dest_mesh[i][0], L);
//                     if ((-right <= x && x < 0) || x > right) { result[i].tran() = -result[i].tran(); }
//                 }
//             return result;
//         } else {
//             FFT::Backward1D fft(3, N, FFT::SYMMETRY_NONE);
//             fft.execute(reinterpret_cast<dcomplex*>(field.data()));
//             field[N] = field[0];
//             RegularMesh3D src_mesh(RegularAxis(left, right, field.size()), RegularAxis(vpos, vpos, 1));
//             return interpolate(src_mesh, field, WrappedMesh<2>(dest_mesh, SOLVER->getGeometry(), true),
//                                defInterpolation<INTERPOLATION_SPLINE>(field_params.method), false);
//         }
//     }
}


}}} // namespace plask
