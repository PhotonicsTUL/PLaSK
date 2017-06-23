// #include "expansioncyl-infini.h"
// #include "solvercyl.h"
// #include "zeros-data.h"
//
// #include "../gauss_legendre.h"
//
// #include <boost/math/special_functions/bessel.hpp>
// #include <boost/math/special_functions/legendre.hpp>
// using boost::math::cyl_bessel_j;
// using boost::math::cyl_bessel_j_zero;
// using boost::math::legendre_p;
//
// #define SOLVER static_cast<BesselSolverCyl*>(solver)
//
// namespace plask { namespace solvers { namespace slab {
//
// ExpansionBesselInfini::ExpansionBesselInfini(BesselSolverCyl* solver): ExpansionBessel(solver)
// {
// }
//
//
// //TODO REMOVE
// void ExpansionBesselInfini::computeBesselZeros()
// {
//     size_t N = SOLVER->size;
//     size_t n = 0;
//     factors.resize(N);
//     if (m < 5) {
//         n = min(N, size_t(100));
//         std::copy_n(bessel_zeros[m], n, factors.begin());
//     }
//     if (n < N) {
//         SOLVER->writelog(LOG_DEBUG, "Computing Bessel function J_({:d}) zeros {:d} to {:d}", m, n+1, N);
//         cyl_bessel_j_zero(double(m), n+1, N-n, factors.begin()+n);
//     }
// }
//
//
//
//
// void ExpansionBesselInfini::init2()
// {
//     SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);
//     computeBesselZeros();
//
//     size_t nseg = rbounds.size() - 1;
//
//     // Estimate necessary number of integration points
//     double k = factors[factors.size()-1];
//
//     double expected = cyl_bessel_j(m+1, k) * rbounds[rbounds.size()-1];
//     expected = 0.5 * expected*expected;
//
//     k /= rbounds[rbounds.size()-1];
//
//     double max_error = SOLVER->integral_error * expected / nseg;
//     double error = 0.;
//
//     std::deque<std::vector<double>> abscissae_cache;
//     std::deque<DataVector<double>> weights_cache;
//
//     auto raxis = plask::make_shared<OrderedAxis>();
//     OrderedAxis::WarningOff nowarn_raxis(raxis);
//
//     double expcts = 0.;
//     for (size_t i = 0; i < nseg; ++i) {
//         double b = rbounds[i+1];
//
//         // excpected value is the second Lommel's integral
//         double expct = expcts;
//         expcts = cyl_bessel_j(m, k*b); expcts = 0.5 * b*b * (expcts*expcts - cyl_bessel_j(m-1, k*b) * cyl_bessel_j(m+1, k*b));
//         expct = expcts - expct;
//
//         double err = 2 * max_error;
//         std::vector<double> points;
//         size_t j, n = 0;
//         double sum;
//         for (j = 0; err > max_error && n <= SOLVER->max_itegration_points; ++j) {
//             n = 4 * (j+1) - 1;
//             if (j == abscissae_cache.size()) {
//                 abscissae_cache.push_back(std::vector<double>());
//                 weights_cache.push_back(DataVector<double>());
//                 gaussLegendre(n, abscissae_cache.back(), weights_cache.back());
//             }
//             assert(j < abscissae_cache.size());
//             assert(j < weights_cache.size());
//             const std::vector<double>& abscissae = abscissae_cache[j];
//             points.clear(); points.reserve(abscissae.size());
//             sum = 0.;
//             for (size_t a = 0; a != abscissae.size(); ++a) {
//                 double r = segments[i].Z + segments[i].D * abscissae[a];
//                 double Jm = cyl_bessel_j(m, k*r);
//                 sum += weights_cache[j][a] * Jm*Jm*r;
//                 points.push_back(r);
//             }
//             sum *= segments[i].D;
//             err = abs(sum - expct);
//         }
//         error += err;
//         raxis->addOrderedPoints(points.begin(), points.end());
//         segments[i].weights = weights_cache[j-1];
//     }
//
//     SOLVER->writelog(LOG_DETAIL, "Sampling structure in {:d} points (error: {:g}/{:g})", raxis->size(), error/expected, SOLVER->integral_error);
//
//     // Allocate memory for integrals
//     size_t nlayers = solver->lcount;
//     layers_integrals.resize(nlayers);
//     iepsilons.resize(nlayers);
//     for (size_t l = 0, nr = raxis->size(); l != nlayers; ++l)
//         iepsilons[l].reset(nr);
//
//     mesh = plask::make_shared<RectangularMesh<2>>(raxis, solver->verts, RectangularMesh<2>::ORDER_01);
//
//     m_changed = false;
// }
//
//
// void ExpansionBesselInfini::reset()
// {
//     layers_integrals.clear();
//     ExpansionBessel::reset();
// }
//
//
// void ExpansionBesselInfini::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH)
// {
//     size_t N = SOLVER->size;
//     dcomplex ik0 = 1. / k0;
//     double b = rbounds[rbounds.size()-1];
//
//     const Integrals& eps = layers_integrals[layer];
//
//     std::fill(RH.begin(), RH.end(), 0.);
//     for (size_t i = 0; i != N; ++i) {
//         size_t is = idxs(i); size_t ip = idxp(i);
//         double i2eta = 1. / (cyl_bessel_j(m+1, factors[i]) * b); i2eta *= i2eta;
//         dcomplex i2etak0 = i2eta * ik0;
//         for (size_t j = 0; j != N; ++j) {
//             size_t jp = idxp(j);
//             double k = factors[j] / b;
//             RH(is, jp) = - i2etak0 * k * (k * (eps.Vmm(i,j) - eps.Vpp(i,j)) + eps.Dm(i,j) + eps.Dp(i,j));
//             RH(ip, jp) = - i2etak0 * k * (k * (eps.Vmm(i,j) + eps.Vpp(i,j)) + eps.Dm(i,j) - eps.Dp(i,j));
//         }
//         RH(is, is)  = k0;
//         RH(ip, ip) += k0;
//     }
//
//     for (size_t i = 0; i != N; ++i) {
//         size_t is = idxs(i); size_t ip = idxp(i);
//         double i2eta = 1. / (cyl_bessel_j(m+1, factors[i]) * b); i2eta *= i2eta;
//         for (size_t j = 0; j != N; ++j) {
//             size_t js = idxs(j); size_t jp = idxp(j);
//             RE(is, js) = i2eta * k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) + eps.Tpp(i,j) + eps.Tpm(i,j));
//             RE(ip, js) = i2eta * k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) - eps.Tpp(i,j) - eps.Tpm(i,j));
//             RE(is, jp) = i2eta * k0 * (eps.Tmm(i,j) - eps.Tmp(i,j) - eps.Tpp(i,j) + eps.Tpm(i,j));
//             RE(ip, jp) = i2eta * k0 * (eps.Tmm(i,j) - eps.Tmp(i,j) + eps.Tpp(i,j) - eps.Tpm(i,j));
//         }
//         double g = factors[i] / b;
//         RE(is, is) -= ik0 * g * g;
//     }
// }
//
// #ifndef NDEBUG
// cmatrix ExpansionBesselInfini::epsVmm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Vmm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselInfini::epsVpp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Vpp(i,j);
//     return result;
// }
// cmatrix ExpansionBesselInfini::epsTmm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tmm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselInfini::epsTpp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tpp(i,j);
//     return result;
// }
// cmatrix ExpansionBesselInfini::epsTmp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tmp(i,j);
//     return result;
// }
// cmatrix ExpansionBesselInfini::epsTpm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tpm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselInfini::epsDm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Dm(i,j);
//     return result;
// }
// cmatrix ExpansionBesselInfini::epsDp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Dp(i,j);
//     return result;
// }
// #endif

// }}} // # namespace plask::solvers::slab
