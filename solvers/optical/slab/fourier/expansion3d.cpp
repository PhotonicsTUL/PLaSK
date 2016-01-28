#include <boost/algorithm/clamp.hpp>
using boost::algorithm::clamp;

#include "expansion3d.h"
#include "solver3d.h"
#include "../meshadapter.h"

#define SOLVER static_cast<FourierSolver3D*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionPW3D::ExpansionPW3D(FourierSolver3D* solver): Expansion(solver), initialized(false),
    symmetry_long(E_UNSPECIFIED), symmetry_tran(E_UNSPECIFIED) {}

size_t ExpansionPW3D::lcount() const {
    return SOLVER->getLayersPoints().size();
}

void ExpansionPW3D::init()
{
    auto geometry = SOLVER->getGeometry();

    periodic_long = geometry->isPeriodic(Geometry3D::DIRECTION_LONG);
    periodic_tran = geometry->isPeriodic(Geometry3D::DIRECTION_TRAN);

    back = geometry->getChild()->getBoundingBox().lower[0];
    front = geometry->getChild()->getBoundingBox().upper[0];
    left = geometry->getChild()->getBoundingBox().lower[1];
    right = geometry->getChild()->getBoundingBox().upper[1];

    size_t refl = SOLVER->refine_long, reft = SOLVER->refine_tran, Ml, Mt;
    if (refl == 0) refl = 1;  if (reft == 0) reft = 1;

    if (symmetry_long != E_UNSPECIFIED && !geometry->isSymmetric(Geometry3D::DIRECTION_LONG))
            throw BadInput(solver->getId(), "Longitudinal symmetry not allowed for asymmetric structure");
    if (symmetry_tran != E_UNSPECIFIED && !geometry->isSymmetric(Geometry3D::DIRECTION_TRAN))
            throw BadInput(solver->getId(), "Transverse symmetry not allowed for asymmetric structure");

    if (geometry->isSymmetric(Geometry3D::DIRECTION_LONG)) {
        if (front <= 0.) {
            back = -back; front = -front;
            std::swap(back, front);
        }
        if (back != 0.) throw BadMesh(SOLVER->getId(), "Longitudinally symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_long()) back = -front;
    }
    if (geometry->isSymmetric(Geometry3D::DIRECTION_TRAN)) {
        if (right <= 0.) {
            left = -left; right = -right;
            std::swap(left, right);
        }
        if (left != 0.) throw BadMesh(SOLVER->getId(), "Transversely symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_tran()) left = -right;
    }

    if (!periodic_long) {
        if (SOLVER->getLongSize() == 0)
            throw BadInput(solver->getId(), "Flat structure in longitudinal direction (size_long = 0) allowed only for periodic geometry");
        // Add PMLs
        if (!symmetric_long()) back -= SOLVER->pml_long.size + SOLVER->pml_long.dist;
        front += SOLVER->pml_long.size + SOLVER->pml_long.dist;
    }
    if (!periodic_tran) {
        if (SOLVER->getTranSize() == 0)
            throw BadInput(solver->getId(), "Flat structure in transverse direction (size_tran = 0) allowed only for periodic geometry");
        // Add PMLs
        if (!symmetric_tran()) left -= SOLVER->pml_tran.size + SOLVER->pml_tran.dist;
        right += SOLVER->pml_tran.size + SOLVER->pml_tran.dist;
    }

    double Ll, Lt;

    if (!symmetric_long()) {
        Ll = front - back;
        Nl = 2 * SOLVER->getLongSize() + 1;
        nNl = 4 * SOLVER->getLongSize() + 1;
        nMl = size_t(round(SOLVER->oversampling_long * nNl));
        Ml = refl * nMl;
        double dx = 0.5 * Ll * (refl-1) / Ml;
        long_mesh = RegularAxis(back-dx, front-dx-Ll/Ml, Ml);
    } else {
        Ll = 2 * front;
        Nl = SOLVER->getLongSize() + 1;
        nNl = 2 * SOLVER->getLongSize() + 1;
        nMl = size_t(round(SOLVER->oversampling_long * nNl));
        Ml = refl * nMl;
        if (SOLVER->dct2()) {
            double dx = 0.25 * Ll / Ml;
            long_mesh = RegularAxis(dx, front-dx, Ml);
        } else {
            size_t nNa = 4 * SOLVER->getLongSize() + 1;
            double dx = 0.5 * Ll * (refl-1) / (refl*nNa);
            long_mesh = RegularAxis(-dx, front+dx, Ml);
        }
    }                                                           // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric_tran()) {                                    //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        Lt = right - left;                                      //  ^ ^ ^ ^ ^
        Nt = 2 * SOLVER->getTranSize() + 1;                     // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nNt = 4 * SOLVER->getTranSize() + 1;
        nMt = size_t(round(SOLVER->oversampling_tran * nNt));   // N = 3  nN = 5  refine = 4  M = 20
        Mt = reft * nMt;                                        // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        double dx = 0.5 * Lt * (reft-1) / Mt;                   //  ^ ^ ^ ^
        tran_mesh = RegularAxis(left-dx, right-dx-Lt/Mt, Mt);   // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    } else {
        Lt = 2 * right;                                         // N = 3  nN = 5  refine = 4  M = 20
        Nt = SOLVER->getTranSize() + 1;                         // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        nNt = 2 * SOLVER->getTranSize() + 1;                    //  ^ ^ ^ ^
        nMt = size_t(round(SOLVER->oversampling_tran * nNt));   // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        Mt = reft * nMt;
        if (SOLVER->dct2()) {
            double dx = 0.25 * Lt / Mt;
            tran_mesh = RegularAxis(dx, right-dx, Mt);
        } else {
            size_t nNa = 4 * SOLVER->getTranSize() + 1;
            double dx = 0.5 * Lt * (reft-1) / (reft*nNa);
            tran_mesh = RegularAxis(-dx, right+dx, Mt);
        }
    }

    if (nMl < nNl || nMt < nNt) throw BadInput(solver->getId(), "Oversampling cannot be smaller than 1");

    SOLVER->writelog(LOG_DETAIL, "Creating expansion{3} with {0}x{1} plane-waves (matrix size: {2})", Nl, Nt, matrixSize(),
                     (!symmetric_long() && !symmetric_tran())? "" :
                     (symmetric_long() && symmetric_tran())? " symmetric in longitudinal and transverse directions" :
                     (!symmetric_long() && symmetric_tran())? " symmetric in transverse direction" : " symmetric in longitudinal direction"
                    );

    auto dct_symmetry = SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1;

    matFFT = FFT::Forward2D(4, nMl, nMt,
                            symmetric_long()? dct_symmetry : FFT::SYMMETRY_NONE,
                            symmetric_tran()? dct_symmetry : FFT::SYMMETRY_NONE);

    // Compute permeability coefficients
    DataVector<Tensor2<dcomplex>> work;
    if (!periodic_long || !periodic_tran) {
        SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure dimensions: {0}um x {1}um)", Ll, Lt);
        size_t ml = (!periodic_long && nNl != nMl)? nMl : 0,
               mt = (!periodic_tran && nNt != nMt)? nMt : 0;
        size_t lenwork = max(ml, mt);
        if (lenwork != 0) work.reset(lenwork, Tensor2<dcomplex>(0.));
    }
    if (periodic_long) {
        mag_long.reset(nNl, Tensor2<dcomplex>(0.));
        mag_long[0].c00 = 1.; mag_long[0].c11 = 1.; // constant 1
    } else {
        DataVector<Tensor2<dcomplex>> lwork;
        if (nNl != nMl) {
            mag_long.reset(nNl);
            lwork = work;
        } else {
            mag_long.reset(nNl, Tensor2<dcomplex>(0.));
            lwork = mag_long;
        }
        double pb = back + SOLVER->pml_long.size, pf = front - SOLVER->pml_long.size;
        if (symmetric_long()) pib = 0;
        else pib = std::lower_bound(long_mesh.begin(), long_mesh.end(), pb) - long_mesh.begin();
        pif = std::lower_bound(long_mesh.begin(), long_mesh.end(), pf) - long_mesh.begin();
        for (size_t i = 0; i != nMl; ++i) {
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
        FFT::Forward1D(2, nMl, symmetric_long()? dct_symmetry : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(lwork.data()));
        // Copy data to its final destination
        if (nNl != nMl) {
            if (symmetric_long()) {
                std::copy_n(work.begin(), nNl, mag_long.begin());
            } else {
                size_t nn = nNl/2;
                std::copy_n(work.begin(), nn+1, mag_long.begin());
                std::copy_n(work.begin()+nMl-nn, nn, mag_long.begin()+nn+1);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI*M_PI / Ll / Lt;   // (2π/L)² / 4
            for (size_t i = 0; i != nNl; ++i) {
                int k = i; if (!symmetric_long() && k > nNl/2) k -= nNl;
                mag_long[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }
    if (periodic_tran) {
        mag_tran.reset(nNt, Tensor2<dcomplex>(0.));
        mag_tran[0].c00 = 1.; mag_tran[0].c11 = 1.; // constant 1
    } else {
        DataVector<Tensor2<dcomplex>> twork;
        if (nNt != nMt) {
            mag_tran.reset(nNt);
            twork = work;
        } else {
            mag_tran.reset(nNt, Tensor2<dcomplex>(0.));
            twork = mag_tran;
        }
        double pl = left + SOLVER->pml_tran.size, pr = right - SOLVER->pml_tran.size;
        if (symmetric_tran()) pil = 0;
        else pil = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pl) - tran_mesh.begin();
        pir = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pr) - tran_mesh.begin();
        for (size_t i = 0; i != nMt; ++i) {
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
        FFT::Forward1D(2, nNt, symmetric_tran()? dct_symmetry : FFT::SYMMETRY_NONE).execute(reinterpret_cast<dcomplex*>(twork.data()));
        // Copy data to its final destination
        if (nNt != nMt) {
            if (symmetric_tran()) {
                std::copy_n(work.begin(), nNt, mag_tran.begin());
            } else {
                size_t nn = nNt/2;
                std::copy_n(work.begin(), nn+1, mag_tran.begin());
                std::copy_n(work.begin()+nMt-nn, nn, mag_tran.begin()+nn+1);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = M_PI*M_PI / Ll / Lt;   // (2π/L)² / 4
            for (size_t i = 0; i != nNt; ++i) {
                int k = i; if (!symmetric_tran() && k > nNt/2) k -= nNt;
                mag_tran[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = lcount();
    coeffs.resize(nlayers);
    diagonals.assign(nlayers, false);

    initialized = true;
}

void ExpansionPW3D::reset() {
    coeffs.clear();
    initialized = false;
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
    auto axis2 = SOLVER->getLayerPoints(layer);

    const double Lt = right - left, Ll = front - back;
    const size_t refl = (SOLVER->refine_long)? SOLVER->refine_long : 1,
                 reft = (SOLVER->refine_tran)? SOLVER->refine_tran : 1;
    const size_t Ml = refl * nMl,  Mt = reft * nMt;
    size_t nN = nNl * nNt, nM = nMl * nMt;
    const double normlim = min(Ll/nMl, Lt/nMt) * 1e-9;

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {0} (sampled at {1}x{2} points) in thread {3}",
                         layer, Ml, Mt, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {0} (sampled at {1}x{2} points)", layer, Ml, Mt);
    #endif

    if (isnan(lam))
        throw BadInput(SOLVER->getId(), "No wavelength specified: set solver lam0 parameter");
        
    auto mesh = plask::make_shared<RectangularMesh<3>>
                           (plask::make_shared<RegularAxis>(long_mesh),
                            plask::make_shared<RegularAxis>(tran_mesh),
                            axis2,
                            RectangularMesh<3>::ORDER_102);
    double matv = axis2->at(0); // at each point along any vertical axis material is the same

    auto temperature = SOLVER->inTemperature(mesh);

    LazyData<double> gain;
    bool gain_connected = SOLVER->inGain.hasProvider(), gain_computed = false;

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    // Make space for the result
    bool oversampled = nNl != nMl || nNt != nMt;
    DataVector<Tensor3<dcomplex>> work;
    if (oversampled) {
        coeffs[layer].reset(nN);
        work.reset(nM, Tensor3<dcomplex>(0.));
    } else {
        coeffs[layer].reset(nN, Tensor3<dcomplex>(0.));
        work = coeffs[layer];
    }

    // Average material parameters
    DataVector<Tensor3<dcomplex>> cell(refl*reft);
    double nfact = 1. / cell.size();

    double pb = back + SOLVER->pml_long.size, pf = front - SOLVER->pml_long.size;
    double pl = left + SOLVER->pml_tran.size, pr = right - SOLVER->pml_tran.size;

    for (size_t it = 0; it != nMt; ++it) {
        size_t tbegin = reft * it; size_t tend = tbegin + reft;
        double tran0 = 0.5 * (tran_mesh[tbegin] + tran_mesh[tend-1]);

        for (size_t il = 0; il != nMl; ++il) {
            size_t lbegin = refl * il; size_t lend = lbegin + refl;
            double long0 = 0.5 * (long_mesh[lbegin] + long_mesh[lend-1]);

            // Store epsilons for a single cell and compute surface normal
            Vec<2> norm(0.,0.);
            for (size_t t = tbegin, j = 0; t != tend; ++t) {
                for (size_t l = lbegin; l != lend; ++l, ++j) {
                    auto material = geometry->getMaterial(vec(long_mesh[l], tran_mesh[t], matv));
                    double T = 0.; // average temperature in all vertical points
                    for (size_t v = mesh->index(l, t, 0), end = mesh->index(l, t, axis2->size()); v != end; ++v) T += temperature[v];
                    T /= axis2->size();
                    cell[j] = material->NR(lam, T);
                    if (cell[j].c01 != 0.) {
                        if (symmetric_long() || symmetric_tran()) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
                    }
                    if (gain_connected && solver->lgained[layer]) {
                        auto roles = geometry->getRolesAt(vec(long_mesh[l], tran_mesh[t], matv));
                        if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                            if (!gain_computed) {
                                gain = SOLVER->inGain(mesh, glam);
                                gain_computed = true;
                            }
                            double g = 0.; // average gain in all vertical points
                            for (size_t v = mesh->index(l, t, 0), end = mesh->index(l, t, axis2->size()); v < end; ++v) g += gain[v];
                            double ni = glam * g/axis2->size() * (0.25e-7/M_PI);
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
                            double h = (pb - long_mesh[l]) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                        } else if (l > pif) {
                            double h = (long_mesh[l] - pf) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor-1.)*pow(h, SOLVER->pml_long.order);
                        }
                        cell[j].c00 *= 1./s;
                        cell[j].c11 *= s;
                        cell[j].c22 *= s;
                    }
                    if (!periodic_tran) {
                        dcomplex s = 1.;
                        if (t < pil) {
                            double h = (pl - tran_mesh[t]) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                        } else if (t > pir) {
                            double h = (tran_mesh[t] - pr) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor-1.)*pow(h, SOLVER->pml_tran.order);
                        }
                        cell[j].c00 *= s;
                        cell[j].c11 *= 1./s;
                        cell[j].c22 *= s;
                    }

                    norm += (real(cell[j].c00) + real(cell[j].c11)) * vec(long_mesh[l] - long0, tran_mesh[t] - tran0);
                }
            }

            double a = abs(norm);
            auto& eps = work[nMl * it + il];
            if (a < normlim) {
                // Nothing to average
                eps = cell[cell.size() / 2];
            } else {

                // Compute avg(eps) and avg(eps**(-1))
                Tensor3<dcomplex> ieps(0.);
                for (size_t t = tbegin, j = 0; t != tend; ++t) {
                    for (size_t l = lbegin; l != lend; ++l, ++j) {
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
                for (size_t t = 0; t != nNt; ++t) copy_coeffs_long(layer, work, t, t);
            } else {
                size_t nn = nNt/2;
                for (size_t t = 0; t != nn+1; ++t) copy_coeffs_long(layer, work, t, t);
                for (size_t tw = nMt-nn, tc = nn+1; tw != nMt; ++tw, ++tc) copy_coeffs_long(layer, work, tw, tc);
            }
        }
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4l = M_PI / ((front-back) * (symmetric_long()? 2 : 1)); bb4l *= bb4l; // (2π/Ll)² / 4
            double bb4t = M_PI / ((right-left) * (symmetric_tran()? 2 : 1)); bb4t *= bb4t; // (2π/Lt)² / 4
            for (size_t it = 0; it != nNt; ++it) {
                int kt = it; if (!symmetric_tran() && kt > nNt/2) kt -= nNt;
                for (size_t il = 0; il != nNl; ++il) {
                    int kl = il; if (!symmetric_long() && kl > nNl/2) kl -= nNl;
                    coeffs[layer][nNl*it+il] *= exp(-SOLVER->smooth * (bb4l * kl*kl + bb4t * kt*kt));
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
            const int nt = symmetric_tran()? nNt-1 : nNt/2,
                      nl = symmetric_long()? nNl-1 : nNl/2;
            double Lt = right-left; if (symmetric_tran()) Lt *= 2;
            double Ll = front-back; if (symmetric_long()) Ll *= 2;
            for (int kt = -nt; kt <= nt; ++kt) {
                size_t t = (kt >= 0)? kt : (symmetric_tran())? -kt : kt + nNt;
                const double phast = kt * (dest_mesh->at(i).c1-left) / Lt;
                for (int kl = -nl; kl <= nl; ++kl) {
                    size_t l = (kl >= 0)? kl : (symmetric_long())? -kl : kl + nNl;
                    eps += coeffs[lay][nNl*t+l] * exp(2*M_PI * I * (kl*(dest_mesh->at(i).c0-back) / Ll + phast));
                }
            }
            eps.c22 = 1. / eps.c22;
            eps.sqrt_inplace();
            return eps;
        });
    } else {
        DataVector<Tensor3<dcomplex>> result(dest_mesh->size(), Tensor3<dcomplex>(0.));
        size_t nl = symmetric_long()? nNl : nNl+1, nt = symmetric_tran()? nNt : nNt+1;
        DataVector<Tensor3<dcomplex>> params(nl * nt);
        for (size_t t = 0; t != nNt; ++t) {
            size_t op = nl * t, oc = nNl * t;
            for (size_t l = 0; l != nNl; ++l) {
                params[op+l] = coeffs[lay][oc+l];
            }
        }
        auto dct_symmetry = SOLVER->dct2()? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1;
        FFT::Backward2D(4, nNl, nNt,
                        symmetric_long()? dct_symmetry : FFT::SYMMETRY_NONE,
                        symmetric_tran()? dct_symmetry : FFT::SYMMETRY_NONE,
                        0, nl
                       )
            .execute(reinterpret_cast<dcomplex*>(params.data()));
        shared_ptr<RegularAxis> lcmesh = plask::make_shared<RegularAxis>(), tcmesh = plask::make_shared<RegularAxis>();
        if (symmetric_long()) {
            if (SOLVER->dct2()) {
                double dx = 0.5 * (front-back) / nl;
                lcmesh->reset(back+dx, front-dx, nl);
            } else {
                lcmesh->reset(0., front, nl);
            }
        } else {
            lcmesh->reset(back, front, nl);
            for (size_t t = 0, end = nl*nt, dist = nl-1; t != end; t += nl) params[dist+t] = params[t];
        }
        if (symmetric_tran()) {
            if (SOLVER->dct2()) {
                double dy = 0.5 * right / nt;
                tcmesh->reset(dy, right-dy, nt);
            } else {
                tcmesh->reset(0., right, nt);
            }
        } else {
            tcmesh->reset(left, right, nt);
            for (size_t l = 0, last = nl*(nt-1); l != nl; ++l) params[last+l] = params[l];
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

    dcomplex k0 = SOLVER->k0, klong = SOLVER->klong, ktran = SOLVER->ktran;
    
    int ordl = SOLVER->getLongSize(), ordt = SOLVER->getTranSize();

    char symx = symmetric_long()? 2 * int(symmetry_long) - 3 : 0,
         symy = symmetric_tran()? 2 * int(symmetry_tran) - 3 : 0;
         // +1: Ex+, Ey-, Hx-, Hy+
         //  0: no symmetry
         // -1: Ex-, Ey+, Hx+, Hy-

    assert(!(symx && klong != 0.));
    assert(!(symy && ktran != 0.));

    double Gx = 2.*M_PI / (front-back) * (symx ? 0.5 : 1.),
           Gy = 2.*M_PI / (right-left) * (symy ? 0.5 : 1.);

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
        size_t nl = (syml == E_UNSPECIFIED)? Nl+1 : Nl;
        size_t nt = (symt == E_UNSPECIFIED)? Nt+1 : Nt;
        if (field_interpolation != INTERPOLATION_FOURIER) {
            int df = SOLVER->dct2()? 0 : 4;
            FFT::Symmetry x1, xz2, yz1, y2;
            if (symmetric_long()) { x1 = FFT::Symmetry(3-syml + df); yz1 = FFT::Symmetry(syml + df); }
            else { x1 = yz1 = FFT::SYMMETRY_NONE; }
            if (symmetric_tran()) { xz2 = FFT::Symmetry(3-symt + df); y2 = FFT::Symmetry(symt + df); }
            else { xz2 = y2 = FFT::SYMMETRY_NONE; }
            fft_x = FFT::Backward2D(1, Nl, Nt, x1, xz2, 3, nl);
            fft_y = FFT::Backward2D(1, Nl, Nt, yz1, y2, 3, nl);
            fft_z = FFT::Backward2D(1, Nl, Nt, yz1, xz2, 3, nl);
        }
        field.reset(nl*nt);
    } else {
        if (field_interpolation != INTERPOLATION_FOURIER)
            fft_z = FFT::Backward2D(3, Nl, Nt, FFT::SYMMETRY_NONE, FFT::SYMMETRY_NONE, 3, Nl+1);
        field.reset((Nl+1)*(Nt+1));
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
    Component syml = (which_field == FIELD_E)? symmetry_long : Component(3-symmetry_long);
    Component symt = (which_field == FIELD_E)? symmetry_tran : Component(3-symmetry_tran);

    size_t nl = (syml == E_UNSPECIFIED)? Nl+1 : Nl,
           nt = (symt == E_UNSPECIFIED)? Nt+1 : Nt;

    const dcomplex kx = SOLVER->klong, ky = SOLVER->ktran;

    int ordl = SOLVER->getLongSize(), ordt = SOLVER->getTranSize();

    double bl = 2*M_PI / (front-back) * (symmetric_long()? 0.5 : 1.0),
           bt = 2*M_PI / (right-left) * (symmetric_tran()? 0.5 : 1.0);

    assert(dynamic_pointer_cast<const MeshD<3>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<3>>(level->mesh());
    double vpos = level->vpos();

    int dxl = 0, dyl = 0, dxt = 0, dyt = 0;
    if (field_interpolation != INTERPOLATION_FOURIER) {
        if (symmetric_long()) {
            if (syml == E_TRAN) dxl = 1; else dyl = 1;
            for (size_t t = 0, end = nl*nt; t != end; t += nl) field[nl-1+t] = Vec<3,dcomplex>(0.,0.,0.);
        }
        if (symmetric_tran()) {
            if (symt == E_TRAN) dxt = 1; else dyt = 1;
            for (size_t l = 0, off = nl*(nt-1); l != Nl; ++l) field[off+l] = Vec<3,dcomplex>(0.,0.,0.);
        }
    }

    if (which_field == FIELD_E) {
        for (int it = symmetric_tran()? 0 : -ordt; it <= ordt; ++it) {
            for (int il = symmetric_long()? 0 : -ordl; il <= ordl; ++il) {
                // How expensive is checking conditions in each loop?
                // Fuck it, the code is much more clear this way.
                size_t iex = nl * (((it<0)?Nt+it:it) - dxt) + ((il<0)?Nl+il:il) - dxl;
                size_t iey = nl * (((it<0)?Nt+it:it) - dyt) + ((il<0)?Nl+il:il) - dyl;
                size_t iez = nl * (((it<0)?Nt+it:it) - dxt) + ((il<0)?Nl+il:il) - dyl;
                if (!(it == 0 && dxt) && !(il == 0 && dxl))
                    field[iex].lon() = E[iEx(il,it)];
                if (!(it == 0 && dyt) && !(il == 0 && dyl))
                    field[iey].tran() = E[iEy(il,it)];
                if (!(it == 0 && dxt) && !(il == 0 && dyl)) {
                    field[iez].vert() = 0.;
                    for (int jt = -ordt; jt <= ordt; ++jt)
                        for (int jl = -ordl; jl <= ordl; ++jl) {
                            double fhx = ((jl < 0 && symmetry_long == E_LONG)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_LONG)? -1. : 1);
                            double fhy = ((jl < 0 && symmetry_long == E_TRAN)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_TRAN)? -1. : 1);
                            field[iez].vert() += iepszz(l,il-jl,it-jt) *
                                (  (bl*double(jl)-kx) * fhy*H[iHy(jl,jt)]
                                 + (bt*double(jt)-ky) * fhx*H[iHx(jl,jt)]);
                        }
                    field[iez].vert() /= SOLVER->k0;
                }
            }
        }
    } else { // which_field == FIELD_H
        for (int it = symmetric_tran()? 0 : -ordt; it <= ordt; ++it) {
            for (int il = symmetric_long()? 0 : -ordl; il <= ordl; ++il) {
                size_t ihx = nl * (((it<0)?Nt+it:it) - dxt) + ((il<0)?Nl+il:il) - dxl;
                size_t ihy = nl * (((it<0)?Nt+it:it) - dyt) + ((il<0)?Nl+il:il) - dyl;
                size_t ihz = nl * (((it<0)?Nt+it:it) - dxt) + ((il<0)?Nl+il:il) - dyl;
                if (!(it == 0 && dxt) && !(il == 0 && dxl))
                    field[ihx].lon() = - H[iHx(il,it)];
                if (!(it == 0 && dyt) && !(il == 0 && dyl))
                    field[ihy].tran() = H[iHy(il,it)];
                if (!(it == 0 && dxt) && !(il == 0 && dyl)) {
                    field[ihz].vert() = 0.;
                    for (int jt = -ordt; jt <= ordt; ++jt)
                        for (int jl = -ordl; jl <= ordl; ++jl) {
                            double fex = ((jl < 0 && symmetry_long == E_TRAN)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_TRAN)? -1. : 1);
                            double fey = ((jl < 0 && symmetry_long == E_LONG)? -1. : 1) *
                                         ((jt < 0 && symmetry_tran == E_LONG)? -1. : 1);
                            field[ihz].vert() += imuzz(l,il-jl,it-jt) *
                                (- (bl*double(jl)-kx) * fey*E[iEy(jl,jt)]
                                 + (bt*double(jt)-ky) * fex*E[iEx(jl,jt)]);
                        }
                    field[ihz].vert() /= SOLVER->k0;
                }
            }
        }
    }

    if (field_interpolation == INTERPOLATION_FOURIER) {
        const double lo0 = symmetric_long()? -front : back, hi0 = front,
                     lo1 = symmetric_tran()? -right : left, hi1 = right;
        DataVector<Vec<3,dcomplex>> result(dest_mesh->size());
        double Ll = (symmetric_long()? 2. : 1.) * (front - back),
               Lt = (symmetric_tran()? 2. : 1.) * (right - left);
        dcomplex bl = 2.*M_PI * I / Ll, bt = 2.*M_PI * I / Lt;
        dcomplex ikx = I * kx, iky = I * ky;
        result.reset(dest_mesh->size(), Vec<3,dcomplex>(0.,0.,0.));
        for (int it = -ordt; it <= ordt; ++it) {
            double ftx = 1., fty = 1.;
            size_t iit;
            if (it < 0) {
                if (symmetric_tran()) {
                    if (symt == E_LONG) fty = -1.;
                    else ftx = -1.;
                    iit = nl * (-it);
                } else {
                    iit = nl * (Nt+it);
                }
            } else {
                iit = nl * it;
            }
            dcomplex gt = bt*double(it) - iky;
            for (int il = -ordl; il <= ordl; ++il) {
                double flx = 1., fly = 1.;
                size_t iil;
                if (il < 0) {
                    if (symmetric_long()) {
                        if (syml == E_LONG) fly = -1.;
                        else flx = -1.;
                        iil = -il;
                    } else {
                        iil = Nl + il;
                    }
                } else {
                    iil = il;
                }
                Vec<3,dcomplex> coeff = field[iit + iil];
                coeff.c0 *= ftx * flx;
                coeff.c1 *= fty * fly;
                coeff.c2 *= ftx * fly;
                dcomplex gl = bl*double(il) - ikx;
                for (size_t ip = 0; ip != dest_mesh->size(); ++ip) {
                    auto p = dest_mesh->at(ip);
                    if (!periodic_long) p.c0 = clamp(p.c0, lo0, hi0);
                    if (!periodic_tran) p.c1 = clamp(p.c1, lo1, hi1);
                    result[ip] += coeff * exp(gl * (p.c0-back) + gt * (p.c1-left));
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
                dy = 0.5 * (right-left) / nt;
            } else {
                for (size_t l = 0, off = nl*Nt; l != Nl; ++l) field[off+l] = field[l];
                dy = 0.;
            }
            if (symmetric_long()) {
                dx = 0.5 * (front-back) / nl;
            } else {
                for (size_t t = 0, end = nl*nt; t != end; t += nl) field[Nl+t] = field[t];
                dx = 0.;
            }
            auto src_mesh = plask::make_shared<RectangularMesh<3>>(
                plask::make_shared<RegularAxis>(back+dx, front-dx, nl),
                plask::make_shared<RegularAxis>(left+dy, right-dy, nt),
                plask::make_shared<RegularAxis>(vpos, vpos, 1),
                RectangularMesh<3>::ORDER_210
            );
            auto result = interpolate(src_mesh, field, dest_mesh, field_interpolation,
                                      InterpolationFlags(SOLVER->getGeometry(),
                                                         symmetric_long()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                                         symmetric_tran()? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                                         InterpolationFlags::Symmetry::NO),
                                      false).claim();
            if (symmetric_long()) {
                double Ll = 2. * front;
                if (syml == E_TRAN)
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double x = std::fmod(dest_mesh->at(i)[0], Ll);
                        if ((-front <= x && x < 0) || x > front) { result[i].lon() = -result[i].lon(); result[i].vert() = -result[i].vert(); }
                    }
                else
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double x = std::fmod(dest_mesh->at(i)[0], Ll);
                        if ((-front <= x && x < 0) || x > front) { result[i].tran() = -result[i].tran(); }
                    }
            } else {
                dcomplex ikx = I * kx;
                for (size_t i = 0; i != dest_mesh->size(); ++i)
                    result[i] *= exp(- ikx * dest_mesh->at(i).c0);
            }
            if (symmetric_tran()) {
                double Lt = 2. * right;
                if (symt == E_TRAN)
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double y = std::fmod(dest_mesh->at(i)[1], Lt);
                        if ((-right <= y && y < 0) || y > right) { result[i].lon() = -result[i].lon(); result[i].vert() = -result[i].vert(); }
                    }
                else
                    for (size_t i = 0; i != dest_mesh->size(); ++i) {
                        double y = std::fmod(dest_mesh->at(i)[1], Lt);
                        if ((-right <= y && y < 0) || y > right) { result[i].tran() = -result[i].tran(); }
                    }
            } else {
                dcomplex iky = I * ky;
                for (size_t i = 0; i != dest_mesh->size(); ++i)
                    result[i] *= exp(- iky * dest_mesh->at(i).c1);
            }
            return result;
        } else {
            fft_z.execute(reinterpret_cast<dcomplex*>(field.data()));
            for (size_t l = 0, off = nl*Nt; l != Nl; ++l) field[off+l] = field[l];
            for (size_t t = 0, end = nl*nt; t != end; t += nl) field[Nl+t] = field[t];
            auto src_mesh = plask::make_shared<RectangularMesh<3>>(
                plask::make_shared<RegularAxis>(back, front, nl),
                plask::make_shared<RegularAxis>(left, right, nt),
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


}}} // namespace plask
