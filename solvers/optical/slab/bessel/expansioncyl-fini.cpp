#include "expansioncyl-fini.h"
#include "solvercyl.h"
#include "zeros-data.h"

#include "../gauss_legendre.h"

#include <boost/math/special_functions/bessel.hpp>
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;
using boost::math::legendre_p;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionBesselFini::ExpansionBesselFini(BesselSolverCyl* solver) : ExpansionBessel(solver) {}

void ExpansionBesselFini::computeBesselZeros() {
    size_t N = SOLVER->size;
    size_t n = 0;
    kpts.resize(N);
    if (m < 5) {
        n = min(N, size_t(100));
        std::copy_n(bessel_zeros[m], n, kpts.begin());
    }
    if (n < N) {
        SOLVER->writelog(LOG_DEBUG, "Computing Bessel function J_({:d}) zeros {:d} to {:d}", m, n + 1, N);
        PLASK_NO_CONVERSION_WARNING_BEGIN
        cyl_bessel_j_zero(double(m), n + 1, N - n, kpts.begin() + n);
        PLASK_NO_WARNING_END
    }
    // #ifndef NDEBUG
    //     for (size_t i = 0; i != N; ++i) {
    //         auto report = [i,m,this]()->bool{
    //             std::cerr << "J(" << m << ", " << kpts[i] << ") = " << cyl_bessel_j(m, kpts[i]) << "\n";
    //             return false;
    //         };
    //         assert(is_zero(cyl_bessel_j(m, kpts[i]), 1e-9) || report());
    //     }
    // #endif
}

void ExpansionBesselFini::init2() {
    SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);
    computeBesselZeros();

    init3();

    // Compute integrals for permeability
    auto raxis = mesh->tran();

    size_t nr = raxis->size(), N = SOLVER->size;
    mu_integrals.reset(N);

    double ib = 1. / rbounds[rbounds.size() - 1];

    // Compute mu integrals
    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {

        size_t pmlseg = segments.size() - 1;

        size_t pmli = raxis->size() - segments[pmlseg].weights.size();
        double pmlr = rbounds[pmlseg];

        aligned_unique_ptr<dcomplex> mu_data(aligned_malloc<dcomplex>(nr));
        aligned_unique_ptr<dcomplex> imu_data(aligned_malloc<dcomplex>(nr));

        for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
            if (wi == nw) {
                nw = segments[++seg].weights.size();
                wi = 0;
            }
            double r = raxis->at(ri);
            double w = segments[seg].weights[wi] * segments[seg].D;

            dcomplex mu = 1., imu = 1.;
            if (ri >= pmli) {
                mu = 1. + (SOLVER->pml.factor - 1.) * pow((r - pmlr) / SOLVER->pml.size, SOLVER->pml.order);
                imu = 1. / mu;
            }

            mu_data.get()[ri] = mu * w;
            imu_data.get()[ri] = imu * w;
        }

        integrateParams(mu_integrals, mu_data.get(), imu_data.get(), mu_data.get());

    } else {

        zero_matrix(mu_integrals.Vzz);
        zero_matrix(mu_integrals.Tss);
        zero_matrix(mu_integrals.Tsp);
        zero_matrix(mu_integrals.Tps);
        zero_matrix(mu_integrals.Tpp);
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            mu_integrals.Vzz(i,i) = k;
            mu_integrals.Tss(i,i) = mu_integrals.Tpp(i,i) = 2.;
        }
    }
}

void ExpansionBesselFini::reset() {
    mu_integrals.reset();
    ExpansionBessel::reset();
}

void ExpansionBesselFini::layerIntegrals(size_t layer, double lam, double glam) { integrateLayer(layer, lam, glam, true); }

void ExpansionBesselFini::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) {
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    size_t N = SOLVER->size;
    dcomplex ik0 = 1. / k0;
    double ib = 1. / rbounds[rbounds.size() - 1];

    const Integrals& eps = layers_integrals[layer];
    #define mu mu_integrals

    for (size_t j = 0; j != N; ++j) {
        size_t js = idxs(j), jp = idxp(j);
        // double k = kpts[j] * ib;
        for (size_t i = 0; i != N; ++i) {
            size_t is = idxs(i), ip = idxp(i);
            double g = kpts[i] * ib;
            dcomplex Rzz = ik0 * g * eps.Vzz(i,j);
            RH(is,js) = 0.5 * (  Rzz - k0 * mu.Tss(i,j));
            RH(is,jp) = 0.5 * ( -Rzz - k0 * mu.Tsp(i,j));
            RH(ip,js) = 0.5 * (  Rzz + k0 * mu.Tps(i,j));
            RH(is,js) = 0.5 * ( -Rzz + k0 * mu.Tpp(i,j));
        }
    }

    for (size_t j = 0; j != N; ++j) {
        size_t js = idxs(j), jp = idxp(j);
        // double k = kpts[j] * ib;
        for (size_t i = 0; i != N; ++i) {
            size_t is = idxs(i), ip = idxp(i);
            double g = kpts[i] * ib;
            dcomplex Rzz = ik0 * g * mu.Vzz(i,j);
            RE(ip,js) = 0.5 * (  Rzz + k0 * eps.Tps(i,j));
            RE(ip,jp) = 0.5 * ( -Rzz + k0 * eps.Tpp(i,j));
            RE(is,js) = 0.5 * ( -Rzz - k0 * eps.Tss(i,j));
            RE(is,jp) = 0.5 * (  Rzz - k0 * eps.Tsp(i,j));
        }
    }
    #undef mu
}

double ExpansionBesselFini::integratePoyntingVert(const cvector& E, const cvector& H) {
    // double result = 0.;
    // for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
    //     double eta = cyl_bessel_j(m + 1, kpts[i]) * rbounds[rbounds.size() - 1];
    //     eta = 2 * eta * eta;  // 4 × ½
    //     size_t is = idxs(i);
    //     size_t ip = idxp(i);
    //     result += real(E[is] * conj(H[is]) + E[ip] * conj(H[ip])) * eta;
    // }
    // return 2e-12 * PI * result;  // µm² -> m²
    return 1.;
}

double ExpansionBesselFini::integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) {
    // double result = 0.;
    // double R = rbounds[rbounds.size()-1];
    // double iRk02 = 1. / (R*R * real(k0*conj(k0)));
    // if (which_field == FIELD_E) {
    //     for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
    //         double eta = cyl_bessel_j(m+1, kpts[i]) * R; eta = 2. * eta*eta; // 4 × ½
    //         size_t is = idxs(i);
    //         size_t ip = idxp(i);
    //         result += real(E[is]*conj(E[is]) + E[ip]*conj(E[ip])) * eta;
    //         // Add Ez²
    //         double g4 = 4. * iRk02 * kpts[i];
    //         for (size_t j = 0; j < N; ++j) {
    //             size_t jp = idxp(j);
    //             result += g4*kpts[j] * layers_integrals[l].VV(i,j) * real(H[ip]*conj(H[jp]));
    //         }
    //     }
    // } else {
    //     for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
    //         double eta = cyl_bessel_j(m+1, kpts[i]) * R; eta = eta*eta; // 2 × ½
    //         size_t is = idxs(i);
    //         size_t ip = idxp(i);
    //         result += (2. * (real(H[is]*conj(H[is]) + H[ip]*conj(H[ip]))) + iRk02*kpts[i]*kpts[i] * real(E[is]*conj(E[is]))) *
    //         eta;
    //     }
    // }
    // return 0.5 * 2*PI * result;
}

// #ifndef NDEBUG
// cmatrix ExpansionBesselFini::muVmm() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Vmm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselFini::muVpp() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Vpp(i,j);
//     return result;
// }
// cmatrix ExpansionBesselFini::muTmm() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Tmm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselFini::muTpp() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Tpp(i,j);
//     return result;
// }
// cmatrix ExpansionBesselFini::muTmp() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Tmp(i,j);
//     return result;
// }
// cmatrix ExpansionBesselFini::muTpm() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Tpm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselFini::muDm() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Dm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselFini::muDp() {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.Dp(i,j);
//     return result;
// }
// dmatrix ExpansionBesselFini::muVV() {
//     size_t N = SOLVER->size;
//     dmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = mu_integrals.VV(i,j);
//     return result;
// }
// #endif

}}}  // namespace plask::optical::slab
