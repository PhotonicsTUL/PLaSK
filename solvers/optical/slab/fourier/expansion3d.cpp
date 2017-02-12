#include <boost/algorithm/clamp.hpp>
using boost::algorithm::clamp;

#include "expansion3d.h"
#include "solver3d.h"
#include "../meshadapter.h"

#define SOLVER static_cast<FourierSolver3D*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionPW3D::ExpansionPW3D(FourierSolver3D* solver): Expansion(solver), 
    vec0(1., 0., 0.), vec1(0., 1., 0.),
    symmetry_long(E_UNSPECIFIED), symmetry_tran(E_UNSPECIFIED),
    initialized(false) {}


void ExpansionPW3D::init()
{
    auto geometry = SOLVER->getGeometry();

    RegularAxis long_mesh, tran_mesh;

    periodic_long = geometry->isPeriodic(Geometry3D::DIRECTION_LONG);
    periodic_tran = geometry->isPeriodic(Geometry3D::DIRECTION_TRAN);

    size_t refl = SOLVER->refine_long, reft = SOLVER->refine_tran, Ml, Mt;
    if (refl == 0) refl = 1;  if (reft == 0) reft = 1;

    if (SOLVER->custom_lattice) {
        if (!periodic_long || !periodic_tran)
            throw BadInput(solver->getId(), "Custom lattice allowed only for periodic geometry");
        if (symmetry_long != E_UNSPECIFIED || symmetry_tran != E_UNSPECIFIED)
            throw BadInput(solver->getId(), "Mode symmetry is not allowed with custom lattice");
        lo0 = 0.;
        lo1 = 0.;
        hi0 = sqrt(vec0.c0*vec0.c0 + vec0.c1*vec0.c1);
        hi1 = sqrt(vec1.c0*vec1.c0 + vec1.c1*vec1.c1);

        //TODO: test if the periodicity vectors match geometry periodicity
        
    } else {
        lo0 = geometry->getChild()->getBoundingBox().lower[0];
        hi0 = geometry->getChild()->getBoundingBox().upper[0];
        lo1 = geometry->getChild()->getBoundingBox().lower[1];
        hi1 = geometry->getChild()->getBoundingBox().upper[1];

        if (symmetry_long != E_UNSPECIFIED && !geometry->isSymmetric(Geometry3D::DIRECTION_LONG))
                throw BadInput(solver->getId(), "Longitudinal symmetry not allowed for asymmetric structure");
        if (symmetry_tran != E_UNSPECIFIED && !geometry->isSymmetric(Geometry3D::DIRECTION_TRAN))
                throw BadInput(solver->getId(), "Transverse symmetry not allowed for asymmetric structure");

        if (geometry->isSymmetric(Geometry3D::DIRECTION_LONG)) {
            if (hi0 <= 0.) {
                lo0 = -lo0; hi0 = -hi0;
                std::swap(lo0, hi0);
            }
            if (lo0 != 0.) throw BadMesh(SOLVER->getId(), "Longitudinally symmetric geometry must have one of its sides at symmetry axis");
            if (!symmetric_long()) lo0 = -hi0;
        }
        if (geometry->isSymmetric(Geometry3D::DIRECTION_TRAN)) {
            if (hi1 <= 0.) {
                lo1 = -lo1; hi1 = -hi1;
                std::swap(lo1, hi1);
            }
            if (lo1 != 0.) throw BadMesh(SOLVER->getId(), "Transversely symmetric geometry must have one of its sides at symmetry axis");
            if (!symmetric_tran()) lo1 = -hi1;
        }

        if (!periodic_long) {
            if (SOLVER->getLongSize() == 0)
                throw BadInput(solver->getId(), "Flat structure in longitudinal direction (size_long = 0) allowed only for periodic geometry");
            // Add PMLs
            if (!symmetric_long()) lo0 -= SOLVER->pml_long.size + SOLVER->pml_long.dist;
            hi0 += SOLVER->pml_long.size + SOLVER->pml_long.dist;
        }
        if (!periodic_tran) {
            if (SOLVER->getTranSize() == 0)
                throw BadInput(solver->getId(), "Flat structure in transverse direction (size_tran = 0) allowed only for periodic geometry");
            // Add PMLs
            if (!symmetric_tran()) lo1 -= SOLVER->pml_tran.size + SOLVER->pml_tran.dist;
            hi1 += SOLVER->pml_tran.size + SOLVER->pml_tran.dist;
        }
    }

    double Ll, Lt;

    if (!symmetric_long()) {
        Ll = hi0 - lo0;
        N0 = 2 * SOLVER->getLongSize() + 1;
        nN0 = 4 * SOLVER->getLongSize() + 1;
        nM0 = size_t(round(SOLVER->oversampling_long * nN0));
        Ml = refl * nM0;
        double dx = 0.5 * Ll * (refl-1) / Ml;
        long_mesh = RegularAxis(lo0-dx, hi0-dx-Ll/Ml, Ml);
    } else {
        Ll = 2 * hi0;
        N0 = SOLVER->getLongSize() + 1;
        nN0 = 2 * SOLVER->getLongSize() + 1;
        nM0 = size_t(round(SOLVER->oversampling_long * nN0));
        Ml = refl * nM0;
        if (SOLVER->dct2()) {
            double dx = 0.25 * Ll / Ml;
            long_mesh = RegularAxis(dx, hi0-dx, Ml);
        } else {
            size_t nNa = 4 * SOLVER->getLongSize() + 1;
            double dx = 0.5 * Ll * (refl-1) / (refl*nNa);
            long_mesh = RegularAxis(-dx, hi0+dx, Ml);
        }
    }                                                           // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric_tran()) {                                    //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        Lt = hi1 - lo1;                                      //  ^ ^ ^ ^ ^
        N1 = 2 * SOLVER->getTranSize() + 1;                     // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nN1 = 4 * SOLVER->getTranSize() + 1;
        nM1 = size_t(round(SOLVER->oversampling_tran * nN1));   // N = 3  nN = 5  refine = 4  M = 20
        Mt = reft * nM1;                                        // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        double dx = 0.5 * Lt * (reft-1) / Mt;                   //  ^ ^ ^ ^
        tran_mesh = RegularAxis(lo1-dx, hi1-dx-Lt/Mt, Mt);   // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    } else {
        Lt = 2 * hi1;                                         // N = 3  nN = 5  refine = 4  M = 20
        N1 = SOLVER->getTranSize() + 1;                         // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        nN1 = 2 * SOLVER->getTranSize() + 1;                    //  ^ ^ ^ ^
        nM1 = size_t(round(SOLVER->oversampling_tran * nN1));   // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        Mt = reft * nM1;
        if (SOLVER->dct2()) {
            double dx = 0.25 * Lt / Mt;
            tran_mesh = RegularAxis(dx, hi1-dx, Mt);
        } else {
            size_t nNa = 4 * SOLVER->getTranSize() + 1;
            double dx = 0.5 * Lt * (reft-1) / (reft*nNa);
            tran_mesh = RegularAxis(-dx, hi1+dx, Mt);
        }
    }

    if (nM0 < nN0 || nM1 < nN1) throw BadInput(solver->getId(), "Oversampling cannot be smaller than 1");

    SOLVER->writelog(LOG_DETAIL, "Creating expansion{3} with {0}x{1} plane-waves (matrix size: {2})", N0, N1, matrixSize(),
                     (!symmetric_long() && !symmetric_tran())? "" :
                     (symmetric_long() && symmetric_tran())? " symmetric in longitudinal and transverse directions" :
                     (!symmetric_long() && symmetric_tran())? " symmetric in transverse direction" : " symmetric in longitudinal direction"
                    );

    if (symmetric_long()) SOLVER->writelog(LOG_DETAIL, "Longitudinal symmetry is {0}", (symmetry_long == E_TRAN)? "Etran" : "Elong");
    if (symmetric_tran()) SOLVER->writelog(LOG_DETAIL, "Transverse symmetry is {0}", (symmetry_tran == E_TRAN)? "Etran" : "Elong");
    
    auto dct_symmetry = SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1;

    matFFT = FFT::Forward2D(4, nM0, nM1,
                            symmetric_long()? dct_symmetry : FFT::SYMMETRY_NONE,
                            symmetric_tran()? dct_symmetry : FFT::SYMMETRY_NONE);

    // Compute permeability coefficients
    DataVector<Tensor2<dcomplex>> work;
    if (!periodic_long || !periodic_tran) {
        SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure dimensions: {0}um x {1}um)", Ll, Lt);
        size_t ml = (!periodic_long && nN0 != nM0)? nM0 : 0,
               mt = (!periodic_tran && nN1 != nM1)? nM1 : 0;
        size_t lenwork = max(ml, mt);
        if (lenwork != 0) work.reset(lenwork, Tensor2<dcomplex>(0.));
    }
    if (periodic_long) {
        mag_long.reset(nN0, Tensor2<dcomplex>(0.));
        mag_long[0].c00 = 1.; mag_long[0].c11 = 1.; // constant 1
    } else {
        DataVector<Tensor2<dcomplex>> lwork;
        if (nN0 != nM0) {
            mag_long.reset(nN0);
            lwork = work;
        } else {
            mag_long.reset(nN0, Tensor2<dcomplex>(0.));
            lwork = mag_long;
        }
        double pb = lo0 + SOLVER->pml_long.size, pf = hi0 - SOLVER->pml_long.size;
        if (symmetric_long()) pib = 0;
        else pib = std::lower_bound(long_mesh.begin(), long_mesh.end(), pb) - long_mesh.begin();
        pif = std::lower_bound(long_mesh.begin(), long_mesh.end(), pf) - long_mesh.begin();
        for (size_t i = 0; i != nM0; ++i) {
            for (size_t j = refl*i, end = refl*(i+1); j != end; ++j) {
                dcomplex s = 1.;
                if (j < pib) {
                    double h = (pb - long_mesh[j]) / SOLVER->pml_long.size;
                    s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                } else if (j > pif) {
                    double h = (long_mesh[j] - pf) / SOLVER->pml_long.size;
                    s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                }
                lwork[i] += Tensor2<dcomplex>(s, 1./s);
            }
            lwork[i] /= refl;
        }
        // Compute FFT
        FFT::Forward1D(2, nM0, symmetric_long()? dct_symmetry : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(lwork.data()));
        // Copy data to its final destination
        if (nN0 != nM0) {
            if (symmetric_long()) {
                std::copy_n(work.begin(), nN0, mag_long.begin());
            } else {
                size_t nn = nN0/2;
                std::copy_n(work.begin(), nn+1, mag_long.begin());
                std::copy_n(work.begin()+nM0-nn, nn, mag_long.begin()+nn+1);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI / Ll; bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN0; ++i) {
                int k = i; if (!symmetric_long() && k > nN0/2) k -= nN0;
                mag_long[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }
    if (periodic_tran) {
        mag_tran.reset(nN1, Tensor2<dcomplex>(0.));
        mag_tran[0].c00 = 1.; mag_tran[0].c11 = 1.; // constant 1
    } else {
        DataVector<Tensor2<dcomplex>> twork;
        if (nN1 != nM1) {
            mag_tran.reset(nN1);
            twork = work;
        } else {
            mag_tran.reset(nN1, Tensor2<dcomplex>(0.));
            twork = mag_tran;
        }
        double pl = lo1 + SOLVER->pml_tran.size, pr = hi1 - SOLVER->pml_tran.size;
        if (symmetric_tran()) pil = 0;
        else pil = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pl) - tran_mesh.begin();
        pir = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pr) - tran_mesh.begin();
        for (size_t i = 0; i != nM1; ++i) {
            for (size_t j = reft*i, end = reft*(i+1); j != end; ++j) {
                dcomplex s = 1.;
                if (j < pil) {
                    double h = (pl - tran_mesh[j]) / SOLVER->pml_tran.size;
                    s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                } else if (j > pir) {
                    double h = (tran_mesh[j] - pr) / SOLVER->pml_tran.size;
                    s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                }
                twork[i] += Tensor2<dcomplex>(s, 1./s);
            }
            twork[i] /= reft;
        }
        // Compute FFT
        FFT::Forward1D(2, nN1, symmetric_tran()? dct_symmetry : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(twork.data()));
        // Copy data to its final destination
        if (nN1 != nM1) {
            if (symmetric_tran()) {
                std::copy_n(work.begin(), nN1, mag_tran.begin());
            } else {
                size_t nn = nN1/2;
                std::copy_n(work.begin(), nn+1, mag_tran.begin());
                std::copy_n(work.begin()+nM1-nn, nn, mag_tran.begin()+nn+1);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI / Lt; bb4 *= bb4;   // (2π/L)² / 4
            for (size_t i = 0; i != nN1; ++i) {
                int k = i; if (!symmetric_tran() && k > nN1/2) k -= nN1;
                mag_tran[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = solver->lcount;
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);

    static const Vec<3,double> vec2(0., 0., 1.);
    
    mesh = plask::make_shared<EquilateralMesh3D>
                           (plask::make_shared<RegularAxis>(long_mesh),
                            plask::make_shared<RegularAxis>(tran_mesh),
                            solver->verts, RectangularMesh<3>::ORDER_102,
                            vec0, vec1, vec2);

    initialized = true;
}

void ExpansionPW3D::reset() {
    coeffs.clear();
    initialized = false;
    k0 = klong = ktran = lam0 = NAN;
    mesh.reset();
}



void ExpansionPW3D::prepareIntegrals(double lam, double glam) {
    temperature = SOLVER->inTemperature(mesh);
    gain_connected = SOLVER->inGain.hasProvider();
    if (gain_connected) {
        if (isnan(glam)) glam = lam;
        gain = SOLVER->inGain(mesh, glam);
    }
}

void ExpansionPW3D::cleanupIntegrals(double lam, double glam) {
    temperature.reset();
    gain.reset();
}


template <typename T1, typename T2>
inline static Tensor3<decltype(T1()*T2())> commutator(const Tensor3<T1>& A, const Tensor3<T2>& B) {
    return Tensor3<decltype(T1()*T2())>(
        A.c00 * B.c00 + A.c01 * B.c01,
        A.c01 * B.c01 + A.c11 * B.c11,
        A.c22 * B.c22,
        0.5 * ((A.c00 + A.c11) * B.c01 + A.c01 * (B.c00 + B.c11))
    );
}

void ExpansionPW3D::layerIntegrals(size_t layer, double lam, double glam)
{
    auto geometry = SOLVER->getGeometry();

    auto axis0 = mesh->axis0, axis1 = mesh->axis1;

    const double L1 = hi1 - lo1, L0 = hi0 - lo0;
    const size_t ref0 = (SOLVER->refine_long)? SOLVER->refine_long : 1,
                 ref1 = (SOLVER->refine_tran)? SOLVER->refine_tran : 1;
    const size_t Ml = ref0 * nM0,  Mt = ref1 * nM1;
    size_t nN = nN0 * nN1, nM = nM0 * nM1;
    const double normlim = min(L0/nM0, L1/nM1) * 1e-9;

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {0} (sampled at {1}x{2} points) in thread {3}",
                         layer, Ml, Mt, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {0} (sampled at {1}x{2} points)", layer, Ml, Mt);
    #endif

    if (isnan(lam))
        throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    double matv;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            matv = solver->verts->at(i);
            break;
        }
    }

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    // Make space for the result
    bool oversampled = nN0 != nM0 || nN1 != nM1;
    DataVector<Tensor3<dcomplex>> work;
    if (oversampled) {
        coeffs[layer].reset(nN);
        work.reset(nM, Tensor3<dcomplex>(0.));
    } else {
        coeffs[layer].reset(nN, Tensor3<dcomplex>(0.));
        work = coeffs[layer];
    }

    // Average material parameters
    DataVector<Tensor3<dcomplex>> cell(ref0*ref1);
    double nfact = 1. / cell.size();

    double pb = lo0 + SOLVER->pml_long.size, pf = hi0 - SOLVER->pml_long.size;
    double pl = lo1 + SOLVER->pml_tran.size, pr = hi1 - SOLVER->pml_tran.size;

    for (size_t i1 = 0; i1 != nM1; ++i1) {
        size_t beg1 = ref1 * i1; size_t end1 = beg1 + ref1;
        double shift1 = 0.5 * (axis1->at(beg1) + axis1->at(end1-1));

        for (size_t i0 = 0; i0 != nM0; ++i0) {
            size_t beg0 = ref0 * i0; size_t end0 = beg0 + ref0;
            double shift0 = 0.5 * (axis0->at(beg0) + axis0->at(end0-1));

            // Store epsilons for a single cell and compute surface normal
            Vec<2> norm(0.,0.);
            for (size_t t = beg1, j = 0; t != end1; ++t) {
                for (size_t l = beg0; l != end0; ++l, ++j) {
                    auto material = geometry->getMaterial(vec(axis0->at(l), axis1->at(t), matv));
                    double T = 0.; int n1 = 0;
                    for (size_t k = 0, v = mesh->index(l, t, 0); k != mesh->axis2->size(); ++v, ++k)
                        if (solver->stack[k] == layer) { T += temperature[v]; n1++; }
                    T /= n1;
                    cell[j] = material->NR(lam, T);
                    if (cell[j].c01 != 0.) {
                        if (symmetric_long() || symmetric_tran()) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
                    }
                    if (gain_connected && solver->lgained[layer]) {
                        auto roles = geometry->getRolesAt(vec(axis0->at(l), axis1->at(t), matv));
                        if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                            double g = 0.; int ng = 0;
                            for (size_t k = 0, v = mesh->index(l, t, 0); k != mesh->axis2->size(); ++v, ++k)
                                if (solver->stack[k] == layer) { g += gain[v]; ng++; }
                            double ni = glam * g/ng * (0.25e-7/M_PI);
                            cell[j].c00.imag(ni);
                            cell[j].c11.imag(ni);
                            cell[j].c22.imag(ni);
                            cell[j].c01.imag(0.);
                        }
                    }
                    cell[j].sqr_inplace();  // make epsilon from NR

                    // Add PMLs
                    if (!periodic_long) {
                        dcomplex s = 1.;
                        if (l < pib) {
                            double h = (pb - axis0->at(l)) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                        } else if (l > pif) {
                            double h = (axis0->at(l) - pf) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                        }
                        cell[j].c00 *= 1./s;
                        cell[j].c11 *= s;
                        cell[j].c22 *= s;
                    }
                    if (!periodic_tran) {
                        dcomplex s = 1.;
                        if (t < pil) {
                            double h = (pl - axis1->at(t)) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                        } else if (t > pir) {
                            double h = (axis1->at(t) - pr) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                        }
                        cell[j].c00 *= s;
                        cell[j].c11 *= 1./s;
                        cell[j].c22 *= s;
                    }

                    norm += (real(cell[j].c00) + real(cell[j].c11)) * vec(axis0->at(l) - shift0, axis1->at(t) - shift1);
                }
            }

            double a = abs(norm);
            auto& eps = work[nM0 * i1 + i0];
            if (a < normlim) {
                // Nothing to average
                eps = cell[cell.size() / 2];
            } else {

                // Compute avg(eps) and avg(eps**(-1))
                Tensor3<dcomplex> ieps(0.);
                for (size_t t = beg1, j = 0; t != end1; ++t) {
                    for (size_t l = beg0; l != end0; ++l, ++j) {
                        eps += cell[j];
                        ieps += cell[j].inv();
                    }
                }
                eps *= nfact;
                ieps *= nfact;

                // Average permittivity tensor according to:
                // [ S. G. Johnson and J. D. Joannopoulos, Opt. Express, vol. 8, pp. 173-190 (2001) ]
                norm /= a;
                Tensor3<double> P(norm.c0*norm.c0, norm.c1*norm.c1, 0., norm.c0*norm.c1);
                Tensor3<double> P1(1.-P.c00, 1.-P.c11, 1., -P.c01);
                eps = commutator(P, ieps.inv()) + commutator(P1, eps);
            }
            if (eps.c22 != 0.) eps.c22 = 1./eps.c22;
        }
    }

    // Check if the layer is uniform
    if (periodic_tran && periodic_long) {
        diagonals[layer] = true;
        for (size_t i = 1; i != nM; ++i) {
            Tensor3<dcomplex> diff = work[i] - work[0];
            if (!(is_zero(diff.c00) && is_zero(diff.c11) && is_zero(diff.c22) && is_zero(diff.c01))) {
                diagonals[layer] = false;
                break;
            }
        }
    } else
        diagonals[layer] = false;

    if (diagonals[layer]) {
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
        if (oversampled) coeffs[layer][0] = work[0];
        std::fill(coeffs[layer].begin()+1, coeffs[layer].end(), Tensor3<dcomplex>(0.));
    } else {
        // Perform FFT
        matFFT.execute(reinterpret_cast<dcomplex*>(work.data()));
        // Copy result
        if (oversampled) {
            if (symmetric_tran()) {
                for (size_t t = 0; t != nN1; ++t) copy_coeffs_long(layer, work, t, t);
            } else {
                size_t nn = nN1/2;
                for (size_t t = 0; t != nn+1; ++t) copy_coeffs_long(layer, work, t, t);
                for (size_t tw = nM1-nn, tc = nn+1; tw != nM1; ++tw, ++tc) copy_coeffs_long(layer, work, tw, tc);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb40 = M_PI / ((hi0-lo0) * (symmetric_long()? 2 : 1)); bb40 *= bb40; // (2π/Ll)² / 4
            double bb41 = M_PI / ((hi1-lo1) * (symmetric_tran()? 2 : 1)); bb41 *= bb41; // (2π/Lt)² / 4
            for (size_t i1 = 0; i1 != nN1; ++i1) {
                int k1 = i1; if (!symmetric_tran() && k1 > nN1/2) k1 -= nN1;
                for (size_t i0 = 0; i0 != nN0; ++i0) {
                    int k0 = i0; if (!symmetric_long() && k0 > nN0/2) k0 -= nN0;
                    coeffs[layer][nN0*i1+i0] *= exp(-SOLVER->smooth * (bb40 * k0*k0 + bb41 * k1*k1));
                }
            }
        }
    }
}


LazyData<Tensor3<dcomplex>> ExpansionPW3D::getMaterialNR(size_t lay, const shared_ptr<const typename LevelsAdapter::Level> &level, InterpolationMethod interp)
{
    assert(dynamic_pointer_cast<const MeshD<3>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<3>>(level->mesh());

    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        return LazyData<Tensor3<dcomplex>>(dest_mesh->size(), [this,lay,dest_mesh](size_t i)->Tensor3<dcomplex>{
            Tensor3<dcomplex> eps(0.);
            const int n1 = symmetric_tran()? nN1-1 : nN1/2,
                      n0 = symmetric_long()? nN0-1 : nN0/2;
            double Lt = hi1-lo1; if (symmetric_tran()) Lt *= 2;
            double Ll = hi0-lo0; if (symmetric_long()) Ll *= 2;
            for (int kt = -n1; kt <= n1; ++kt) {
                size_t t = (kt >= 0)? kt : (symmetric_tran())? -kt : kt + nN1;
                const double phast = kt * (dest_mesh->at(i).c1-lo1) / Lt;
                for (int kl = -n0; kl <= n0; ++kl) {
                    size_t l = (kl >= 0)? kl : (symmetric_long())? -kl : kl + nN0;
                    eps += coeffs[lay][nN0*t+l] * exp(2*M_PI * I * (kl*(dest_mesh->at(i).c0-lo0) / Ll + phast));
                }
            }
            eps.c22 = 1. / eps.c22;
            eps.sqrt_inplace();
            return eps;
        });
    } else {
        DataVector<Tensor3<dcomplex>> result(dest_mesh->size(), Tensor3<dcomplex>(0.));
        size_t n0 = symmetric_long()? nN0 : nN0+1, n1 = symmetric_tran()? nN1 : nN1+1;
        DataVector<Tensor3<dcomplex>> params(n0 * n1);
        for (size_t t = 0; t != nN1; ++t) {
            size_t op = n0 * t, oc = nN0 * t;
            for (size_t l = 0; l != nN0; ++l) {
                params[op+l] = coeffs[lay][oc+l];
            }
        }
        auto dct_symmetry = SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1;
        FFT::Backward2D(4, nN0, nN1,
                        symmetric_long()? dct_symmetry : FFT::SYMMETRY_NONE,
                        symmetric_tran()? dct_symmetry : FFT::SYMMETRY_NONE,
                        0, n0
                       )
            .execute(reinterpret_cast<dcomplex*>(params.data()));
        shared_ptr<RegularAxis> lcmesh = plask::make_shared<RegularAxis>(), tcmesh = plask::make_shared<RegularAxis>();
        if (symmetric_long()) {
            if (SOLVER->dct2()) {
                double dx = 0.5 * (hi0-lo0) / n0;
                lcmesh->reset(lo0+dx, hi0-dx, n0);
            } else {
                lcmesh->reset(0., hi0, n0);
            }
        } else {
            lcmesh->reset(lo0, hi0, n0);
            for (size_t t = 0, end = n0*n1, dist = n0-1; t != end; t += n0) params[dist+t] = params[t];
        }
        if (symmetric_tran()) {
            if (SOLVER->dct2()) {
                double dy = 0.5 * hi1 / n1;
                tcmesh->reset(dy, hi1-dy, n1);
            } else {
                tcmesh->reset(0., hi1, n1);
            }
        } else {
            tcmesh->reset(lo1, hi1, n1);
            for (size_t l = 0, last = n0*(n1-1); l != n0; ++l) params[last+l] = params[l];
        }
        for (Tensor3<dcomplex>& eps: params) {
            eps.c22 = 1. / eps.c22;
            eps.sqrt_inplace();
        }
        auto src_mesh = plask::make_shared<RectangularMesh<3>>(lcmesh, tcmesh,
                            plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1), RectangularMesh<3>::ORDER_210);
        return interpolate(src_mesh, params, dest_mesh, interp,
                           InterpolationFlags(SOLVER->getGeometry(),
                                              symmetric_long()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                              symmetric_tran()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                              InterpolationFlags::Symmetry::POSITIVE)
                          );
    }
}


void ExpansionPW3D::getMatrices(size_t lay, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);

    int ordl = SOLVER->getLongSize(), ordt = SOLVER->getTranSize();

    char symx = symmetric_long()? 2 * int(symmetry_long) - 3 : 0,
         symy = symmetric_tran()? 2 * int(symmetry_tran) - 3 : 0;
         // +1: Ex+, Ey-, Hx-, Hy+
         //  0: no symmetry
         // -1: Ex-, Ey+, Hx+, Hy-

    assert(!(symx && klong != 0.));
    assert(!(symy && ktran != 0.));

    assert(!isnan(k0.real()) && !isnan(k0.imag()));

    double Gx = 2.*M_PI / (hi0-lo0) * (symx ? 0.5 : 1.),
           Gy = 2.*M_PI / (hi1-lo1) * (symy ? 0.5 : 1.);

    dcomplex ik0 = 1./k0;

    size_t N = (symx ? ordl+1 : 2*ordl+1) * (symy ? ordt+1 : 2*ordt+1);

    std::fill_n(RE.data(), 4*N*N, dcomplex(0.));
    std::fill_n(RH.data(), 4*N*N, dcomplex(0.));

    for (int iy = (symy ? 0 : -ordt); iy <= ordt; ++iy) {
        dcomplex gy = iy * Gy - ktran;
        for (int ix = (symx ? 0 : -ordl); ix <= ordl; ++ix) {
            dcomplex gx = ix * Gx - klong;
            size_t iex = iEx(ix, iy), iey = iEy(ix, iy);
            size_t ihx = iHx(ix, iy), ihy = iHy(ix, iy);

            for (int jy = -ordt; jy <= ordt; ++jy) {
                dcomplex py = jy * Gy - ktran;
                int ijy = iy - jy; if (symy && ijy < 0) ijy = - ijy;
                for (int jx = -ordl; jx <= ordl; ++jx) {
                    dcomplex px = jx * Gx - klong;
                    int ijx = ix - jx; if (symx && ijx < 0) ijx = - ijx;
                    size_t jex = iEx(jx, jy), jey = iEy(jx, jy);
                    size_t jhx = iHx(jx, jy), jhy = iHy(jx, jy);
                    double fx = 1., fy = 1.;
                    if (symx && jx < 0) { fx *= symx; fy *= -symx; }
                    if (symy && jy < 0) { fx *= symy; fy *= -symy; }
                    dcomplex ieps = iepszz(lay, ijx, ijy) * ik0;
                    RH(iex,jhy) += fx * (- gx * px * ieps + k0 * muyy(lay, ijx, ijy));
                    RH(iey,jhy) += fx * (- gy * px * ieps);
                    RH(iex,jhx) += fy * (- gx * py * ieps);
                    RH(iey,jhx) += fy * (- gy * py * ieps + k0 * muxx(lay, ijx, ijy));
                    dcomplex imu = imuzz(lay, ijx, ijy) * ik0;
                    RE(ihy,jex) += fx * (- gy * py * imu + k0 * epsxx(lay, ijx, ijy));
                    RE(ihx,jex) += fx * (  gx * py * imu + k0 * epsyx(lay, ijx, ijy));
                    RE(ihy,jey) += fy * (  gy * px * imu + k0 * epsxy(lay, ijx, ijy));
                    RE(ihx,jey) += fy * (- gx * px * imu + k0 * epsyy(lay, ijx, ijy));
                }
            }

            // Ugly hack to avoid singularity
            if (RE(iex, iex) == 0.) RE(iex, iex) = 1e-32;
            if (RE(iey, iey) == 0.) RE(iey, iey) = 1e-32;
            if (RH(ihx, ihx) == 0.) RH(ihx, ihx) = 1e-32;
            if (RH(ihy, ihy) == 0.) RH(ihy, ihy) = 1e-32;
        }
    }
}


void ExpansionPW3D::prepareField()
{
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_FOURIER;
    if (symmetric_long() || symmetric_tran()) {
        Component syml = (which_field == FIELD_E)? symmetry_long : Component(3-symmetry_long),
                  symt = (which_field == FIELD_E)? symmetry_tran : Component(3-symmetry_tran);
        size_t n0 = (syml == E_UNSPECIFIED)? N0+1 : N0;
        size_t n1 = (symt == E_UNSPECIFIED)? N1+1 : N1;
        if (field_interpolation != INTERPOLATION_FOURIER) {
            int df = SOLVER->dct2()? 0 : 4;
            FFT::Symmetry x1, xz2, yz1, y2;
            if (symmetric_long()) { x1 = FFT::Symmetry(3-syml + df); yz1 = FFT::Symmetry(syml + df); }
            else { x1 = yz1 = FFT::SYMMETRY_NONE; }
            if (symmetric_tran()) { xz2 = FFT::Symmetry(3-symt + df); y2 = FFT::Symmetry(symt + df); }
            else { xz2 = y2 = FFT::SYMMETRY_NONE; }
            fft_x = FFT::Backward2D(1, N0, N1, x1, xz2, 3, n0);
            fft_y = FFT::Backward2D(1, N0, N1, yz1, y2, 3, n0);
            fft_z = FFT::Backward2D(1, N0, N1, yz1, xz2, 3, n0);
        }
        field.reset(n0*n1);
    } else {
        if (field_interpolation != INTERPOLATION_FOURIER)
            fft_z = FFT::Backward2D(3, N0, N1, FFT::SYMMETRY_NONE, FFT::SYMMETRY_NONE, 3, N0+1);
        field.reset((N0+1)*(N1+1));
    }
}

void ExpansionPW3D::cleanupField()
{
    field.reset();
    fft_x = FFT::Backward2D();
    fft_y = FFT::Backward2D();
    fft_z = FFT::Backward2D();
}

// TODO fields must be carefully verified

LazyData<Vec<3, dcomplex>> ExpansionPW3D::getField(size_t l, const shared_ptr<const typename LevelsAdapter::Level> &level, const cvector& E, const cvector& H)
{
    Component syml = (which_field == FIELD_E)? symmetry_long : Component((3-symmetry_long) % 3);
    Component symt = (which_field == FIELD_E)? symmetry_tran : Component((3-symmetry_tran) % 3);

    size_t n0 = (syml == E_UNSPECIFIED)? N0+1 : N0,
           n1 = (symt == E_UNSPECIFIED)? N1+1 : N1;

    const dcomplex kx = klong, ky = ktran;

    int ord0 = SOLVER->getLongSize(), ord1 = SOLVER->getTranSize();

    double bl = 2*M_PI / (hi0-lo0) * (symmetric_long()? 0.5 : 1.0),
           bt = 2*M_PI / (hi1-lo1) * (symmetric_tran()? 0.5 : 1.0);

    assert(dynamic_pointer_cast<const MeshD<3>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<3>>(level->mesh());
    double vpos = level->vpos();

    int dxl = 0, dyl = 0, dxt = 0, dyt = 0;
    if (field_interpolation != INTERPOLATION_FOURIER) {
        if (symmetric_long()) {
            if (syml == E_TRAN) dxl = 1; else dyl = 1;
            for (size_t t = 0, end = n0*n1; t != end; t += n0) field[n0-1+t] = Vec<3,dcomplex>(0.,0.,0.);
        }
        if (symmetric_tran()) {
            if (symt == E_TRAN) dxt = 1; else dyt = 1;
            for (size_t l = 0, off = n0*(n1-1); l != N0; ++l) field[off+l] = Vec<3,dcomplex>(0.,0.,0.);
        }
    }

    if (which_field == FIELD_E) {
        for (int i1 = symmetric_tran()? 0 : -ord1; i1 <= ord1; ++i1) {
            for (int i0 = symmetric_long()? 0 : -ord0; i0 <= ord0; ++i0) {
                // How expensive is checking conditions in each loop?
                // Fuck it, the code is much more clear this way!
                size_t iex = n0 * (((i1<0)?N1+i1:i1) - dxt) + ((i0<0)?N0+i0:i0) - dxl;
                size_t iey = n0 * (((i1<0)?N1+i1:i1) - dyt) + ((i0<0)?N0+i0:i0) - dyl;
                size_t iez = n0 * (((i1<0)?N1+i1:i1) - dxt) + ((i0<0)?N0+i0:i0) - dyl;
                if (!(i1 == 0 && dxt) && !(i0 == 0 && dxl))
                    field[iex].lon() = E[iEx(i0,i1)];
                if (!(i1 == 0 && dyt) && !(i0 == 0 && dyl))
                    field[iey].tran() = E[iEy(i0,i1)];
                if (!(i1 == 0 && dxt) && !(i0 == 0 && dyl)) {
                    field[iez].vert() = 0.;
                    for (int jt = -ord1; jt <= ord1; ++jt)
                        for (int jl = -ord0; jl <= ord0; ++jl) {
                            double fhx = ((jl < 0 && symmetry_long == E_LONG)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_LONG)? -1. : 1);
                            double fhy = ((jl < 0 && symmetry_long == E_TRAN)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_TRAN)? -1. : 1);
                            field[iez].vert() += iepszz(l,i0-jl,i1-jt) *
                                (  (bl*double(jl)-kx) * fhy*H[iHy(jl,jt)]
                                 + (bt*double(jt)-ky) * fhx*H[iHx(jl,jt)]);
                        }
                    field[iez].vert() /= k0;
                }
            }
        }
    } else { // which_field == FIELD_H
        for (int i1 = symmetric_tran()? 0 : -ord1; i1 <= ord1; ++i1) {
            for (int i0 = symmetric_long()? 0 : -ord0; i0 <= ord0; ++i0) {
                size_t ihx = n0 * (((i1<0)?N1+i1:i1) - dxt) + ((i0<0)?N0+i0:i0) - dxl;
                size_t ihy = n0 * (((i1<0)?N1+i1:i1) - dyt) + ((i0<0)?N0+i0:i0) - dyl;
                size_t ihz = n0 * (((i1<0)?N1+i1:i1) - dxt) + ((i0<0)?N0+i0:i0) - dyl;
                if (!(i1 == 0 && dxt) && !(i0 == 0 && dxl))
                    field[ihx].lon() = - H[iHx(i0,i1)];
                if (!(i1 == 0 && dyt) && !(i0 == 0 && dyl))
                    field[ihy].tran() = H[iHy(i0,i1)];
                if (!(i1 == 0 && dxt) && !(i0 == 0 && dyl)) {
                    field[ihz].vert() = 0.;
                    for (int jt = -ord1; jt <= ord1; ++jt)
                        for (int jl = -ord0; jl <= ord0; ++jl) {
                            double fex = ((jl < 0 && symmetry_long == E_TRAN)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_TRAN)? -1. : 1);
                            double fey = ((jl < 0 && symmetry_long == E_LONG)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_LONG)? -1. : 1);
                            field[ihz].vert() += imuzz(l,i0-jl,i1-jt) *
                                (- (bl*double(jl)-kx) * fey*E[iEy(jl,jt)]
                                 + (bt*double(jt)-ky) * fex*E[iEx(jl,jt)]);
                        }
                    field[ihz].vert() /= k0;
                }
            }
        }
    }

    if (field_interpolation == INTERPOLATION_FOURIER) {
        const double lo0 = symmetric_long()? -this->hi0 : this->lo0,
                     lo1 = symmetric_tran()? -this->hi1 : this->lo1;
        DataVector<Vec<3,dcomplex>> result(dest_mesh->size());
        double Ll = (symmetric_long()? 2. : 1.) * (this->hi0 - this->lo0),
               Lt = (symmetric_tran()? 2. : 1.) * (this->hi1 - this->lo1);
        dcomplex bl = 2.*M_PI * I / Ll, bt = 2.*M_PI * I / Lt;
        dcomplex ikx = I * kx, iky = I * ky;
        result.reset(dest_mesh->size(), Vec<3,dcomplex>(0.,0.,0.));
        for (int i1 = -ord1; i1 <= ord1; ++i1) {
            double ftx = 1., fty = 1.;
            size_t iit;
            if (i1 < 0) {
                if (symmetric_tran()) {
                    if (symt == E_LONG) fty = -1.;
                    else ftx = -1.;
                    iit = n0 * (-i1);
                } else {
                    iit = n0 * (N1+i1);
                }
            } else {
                iit = n0 * i1;
            }
            dcomplex gt = bt*double(i1) - iky;
            for (int i0 = -ord0; i0 <= ord0; ++i0) {
                double flx = 1., fly = 1.;
                size_t iil;
                if (i0 < 0) {
                    if (symmetric_long()) {
                        if (syml == E_LONG) fly = -1.;
                        else flx = -1.;
                        iil = -i0;
                    } else {
                        iil = N0 + i0;
                    }
                } else {
                    iil = i0;
                }
                Vec<3,dcomplex> coeff = field[iit + iil];
                coeff.c0 *= ftx * flx;
                coeff.c1 *= fty * fly;
                coeff.c2 *= ftx * fly;
                dcomplex gl = bl*double(i0) - ikx;
                for (size_t ip = 0; ip != dest_mesh->size(); ++ip) {
                    auto p = dest_mesh->at(ip);
                    if (!periodic_long) p.c0 = clamp(p.c0, lo0, hi0);
                    if (!periodic_tran) p.c1 = clamp(p.c1, lo1, hi1);
                    result[ip] += coeff * exp(gl * (p.c0-this->lo0) + gt * (p.c1-this->lo1));
                }
            }
        }
        return result;
    } else {
        if (symmetric_long() || symmetric_tran()) {
            fft_x.execute(&(field.data()->lon()));
            fft_y.execute(&(field.data()->tran()));
            fft_z.execute(&(field.data()->vert()));
            double dx, dy;
            if (symmetric_tran()) {
                dy = 0.5 * (hi1-lo1) / n1;
            } else {
                for (size_t l = 0, off = n0*N1; l != N0; ++l) field[off+l] = field[l];
                dy = 0.;
            }
            if (symmetric_long()) {
                dx = 0.5 * (hi0-lo0) / n0;
            } else {
                for (size_t t = 0, end = n0*n1; t != end; t += n0) field[N0+t] = field[t];
                dx = 0.;
            }
            auto src_mesh = plask::make_shared<RectangularMesh<3>>(
                plask::make_shared<RegularAxis>(lo0+dx, hi0-dx, n0),
                plask::make_shared<RegularAxis>(lo1+dy, hi1-dy, n1),
                plask::make_shared<RegularAxis>(vpos, vpos, 1),
                RectangularMesh<3>::ORDER_210
            );
            auto result = interpolate(src_mesh, field, dest_mesh, field_interpolation,
                                      InterpolationFlags(SOLVER->getGeometry(),
                                                         symmetric_long()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                                         symmetric_tran()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                                         InterpolationFlags::Symmetry::NO),
                                      false).claim();
            if (symmetric_long()) {  // manually change sign where the field has negative symmetry
                double Ll = 2. * hi0;
                if (syml == E_TRAN)
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double x = std::fmod(dest_mesh->at(i)[0], Ll);
                        if ((-hi0 <= x && x < 0) || x > hi0) { result[i].lon() = -result[i].lon(); result[i].vert() = -result[i].vert(); }
                    }
                else
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double x = std::fmod(dest_mesh->at(i)[0], Ll);
                        if ((-hi0 <= x && x < 0) || x > hi0) { result[i].tran() = -result[i].tran(); }
                    }
            } else {
                dcomplex ikx = I * kx;
                for (size_t i = 0; i != dest_mesh->size(); ++i)
                    result[i] *= exp(- ikx * dest_mesh->at(i).c0);
            }
            if (symmetric_tran()) {  // manually change sign where the field has negative symmetry
                double Lt = 2. * hi1;
                if (symt == E_TRAN)
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double y = std::fmod(dest_mesh->at(i)[1], Lt);
                        if ((-hi1 <= y && y < 0) || y > hi1) { result[i].lon() = -result[i].lon(); result[i].vert() = -result[i].vert(); }
                    }
                else
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double y = std::fmod(dest_mesh->at(i)[1], Lt);
                        if ((-hi1 <= y && y < 0) || y > hi1) { result[i].tran() = -result[i].tran(); }
                    }
            } else {
                dcomplex iky = I * ky;
                for (size_t i = 0; i != dest_mesh->size(); ++i)
                    result[i] *= exp(- iky * dest_mesh->at(i).c1);
            }
            return result;
        } else {
            fft_z.execute(reinterpret_cast<dcomplex*>(field.data()));
            for (size_t l = 0, off = n0*N1; l != N0; ++l) field[off+l] = field[l];
            for (size_t t = 0, end = n0*n1; t != end; t += n0) field[N0+t] = field[t];
            auto src_mesh = plask::make_shared<RectangularMesh<3>>(
                plask::make_shared<RegularAxis>(lo0, hi0, n0),
                plask::make_shared<RegularAxis>(lo1, hi1, n1),
                plask::make_shared<RegularAxis>(vpos, vpos, 1),
                RectangularMesh<3>::ORDER_210
            );
            auto result = interpolate(src_mesh, field, dest_mesh, field_interpolation,
                                      InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::NO, InterpolationFlags::Symmetry::NO, InterpolationFlags::Symmetry::NO),
                                      false).claim();
            dcomplex ikx = I * kx, iky = I * ky;
            for (size_t i = 0; i != dest_mesh->size(); ++i)
                result[i] *= exp(- ikx * dest_mesh->at(i).c0 - iky * dest_mesh->at(i).c1);
            return result;
        }
    }
}


double ExpansionPW3D::integratePoyntingVert(const cvector& E, const cvector& H)
{
    double P = 0.;

    int ordl = SOLVER->getLongSize(), ordt = SOLVER->getTranSize();

    for (int iy = -ordt; iy <= ordt; ++iy) {
        for (int ix = -ordl; ix <= ordl; ++ix) {
            P += real(E[iEx(ix,iy)] * conj(H[iHy(ix,iy)]) - E[iEy(ix,iy)] * conj(H[iHx(ix,iy)]));
        }
    }

    return P * (hi0 - lo0) * (hi1 - lo1) * 1e-12; // µm² -> m²
}


}}} // namespace plask
