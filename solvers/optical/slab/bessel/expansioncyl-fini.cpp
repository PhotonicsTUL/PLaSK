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
        cyl_bessel_j_zero(double(m), n+1, N-n, kpts.begin()+n);
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

    double ib = 1. / rbounds[rbounds.size() - 1];

    // Compute mu integrals
    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {

        SOLVER->writelog(LOG_DETAIL, "Computing permeability integrals with {} rule", SOLVER->ruleName());

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

        switch (SOLVER->rule) {
          case BesselSolverCyl::RULE_INVERSE_1:
          case BesselSolverCyl::RULE_INVERSE_2:
          case BesselSolverCyl::RULE_INVERSE_3:
            integrateParams(mu_integrals, mu_data.get(), mu_data.get(), mu_data.get(), 1., 1., 1.); break;
          case BesselSolverCyl::RULE_SEMI_INVERSE:
            integrateParams(mu_integrals, mu_data.get(), imu_data.get(), mu_data.get(), 1., 1., 1.); break;
          case BesselSolverCyl::RULE_DIRECT:
            integrateParams(mu_integrals, mu_data.get(), imu_data.get(), imu_data.get(), 1., 1., 1.); break;
        }

    } else {

        mu_integrals.reset(N);
        zero_matrix(mu_integrals.V_k);
        zero_matrix(mu_integrals.Tss);
        zero_matrix(mu_integrals.Tsp);
        zero_matrix(mu_integrals.Tps);
        zero_matrix(mu_integrals.Tpp);
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            mu_integrals.V_k(i,i) = k;
            mu_integrals.Tss(i,i) = mu_integrals.Tpp(i,i) = 2.;
        }
    }
}

void ExpansionBesselFini::reset() {
    mu_integrals.reset();
    ExpansionBessel::reset();
}

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
            dcomplex gVk = ik0 * g * eps.V_k(i,j);
            RH(is,jp) = 0.5 * (  gVk - k0 * mu.Tsp(i,j) );
            RH(is,js) = 0.5 * (  gVk - k0 * mu.Tss(i,j) );
            RH(ip,jp) = 0.5 * ( -gVk + k0 * mu.Tpp(i,j) );
            RH(ip,js) = 0.5 * ( -gVk + k0 * mu.Tps(i,j) );
        }
    }

    for (size_t j = 0; j != N; ++j) {
        size_t js = idxs(j), jp = idxp(j);
        // double k = kpts[j] * ib;
        for (size_t i = 0; i != N; ++i) {
            size_t is = idxs(i), ip = idxp(i);
            double g = kpts[i] * ib;
            dcomplex gVk = ik0 * g * mu.V_k(i,j);
            RE(ip,js) = 0.5 * ( -gVk + k0 * eps.Tps(i,j) );
            RE(ip,jp) = 0.5 * ( -gVk + k0 * eps.Tpp(i,j) );
            RE(is,js) = 0.5 * (  gVk - k0 * eps.Tss(i,j) );
            RE(is,jp) = 0.5 * (  gVk - k0 * eps.Tsp(i,j) );
        }
    }
    #undef mu
}

double ExpansionBesselFini::fieldFactor(size_t i) {
    double eta = cyl_bessel_j(m + 1, kpts[i]) * rbounds[rbounds.size() - 1];
    return 0.5 * eta * eta;
}


#ifndef NDEBUG
cmatrix ExpansionBesselFini::muV_k() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.V_k(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTss() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tss(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTsp() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tsp(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTps() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tps(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTpp() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tpp(i,j);
    return result;
}
#endif

}}}  // namespace plask::optical::slab
