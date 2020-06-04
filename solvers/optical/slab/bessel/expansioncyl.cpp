#include "expansioncyl.h"
#include "solvercyl.h"
#include "zeros-data.h"

#include "../gauss_legendre.h"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;
using boost::math::legendre_p;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionBessel::ExpansionBessel(BesselSolverCyl* solver) : Expansion(solver), m(1), initialized(false), m_changed(true) {}

size_t ExpansionBessel::matrixSize() const {
    return 2 * SOLVER->size;  // TODO should be N for m = 0?
}

void ExpansionBessel::init1() {
    // Initialize segments
    if (SOLVER->mesh)
        rbounds = OrderedAxis(*SOLVER->getMesh());
    else
        rbounds = std::move(*makeGeometryGrid1D(SOLVER->getGeometry()));
    rbounds.addPoint(0.);
    OrderedAxis::WarningOff nowarn_rbounds(rbounds);
    size_t nseg = rbounds.size() - 1;
    if (SOLVER->pml.dist > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.dist);
    if (SOLVER->pml.size > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.size);
    segments.resize(nseg);
    double a, b = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        a = b;
        b = rbounds[i + 1];
        segments[i].Z = 0.5 * (a + b);
        segments[i].D = 0.5 * (b - a);
    }

    diagonals.assign(solver->lcount, false);
    initialized = true;
    m_changed = true;
}

void ExpansionBessel::reset() {
    layers_integrals.clear();
    segments.clear();
    kpts.clear();
    initialized = false;
    mesh.reset();
    temporary.reset();
}

void ExpansionBessel::init3() {
    size_t nseg = rbounds.size() - 1;

    // Estimate necessary number of integration points
    double k = kpts[kpts.size() - 1];

    double expected = cyl_bessel_j(m+1, k) * rbounds[rbounds.size() - 1];
    expected = 0.5 * expected * expected;

    k /= rbounds[rbounds.size() - 1];

    double max_error = SOLVER->integral_error * expected / double(nseg);
    double error = 0.;

    std::deque<std::vector<double>> abscissae_cache;
    std::deque<DataVector<double>> weights_cache;

    auto raxis = plask::make_shared<OrderedAxis>();
    OrderedAxis::WarningOff nowarn_raxis(raxis);

    double expcts = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        double b = rbounds[i + 1];

        // excpected value is the second Lommel's integral
        double expct = expcts;
        expcts = cyl_bessel_j(m, k*b);
        expcts = 0.5 * b * b * (expcts * expcts - cyl_bessel_j(m-1, k*b) * cyl_bessel_j(m+1, k*b));
        expct = expcts - expct;

        double err = 2 * max_error;
        std::vector<double> points;
        size_t j, n = 0;
        double sum;
        for (j = 0; err > max_error && n <= SOLVER->max_integration_points; ++j) {
            n = 4 * (j + 1) - 1;
            if (j == abscissae_cache.size()) {
                abscissae_cache.push_back(std::vector<double>());
                weights_cache.push_back(DataVector<double>());
                gaussLegendre(n, abscissae_cache.back(), weights_cache.back());
            }
            assert(j < abscissae_cache.size());
            assert(j < weights_cache.size());
            const std::vector<double>& abscissae = abscissae_cache[j];
            points.clear();
            points.reserve(abscissae.size());
            sum = 0.;
            for (size_t a = 0; a != abscissae.size(); ++a) {
                double r = segments[i].Z + segments[i].D * abscissae[a];
                double Jm = cyl_bessel_j(m, k*r);
                sum += weights_cache[j][a] * Jm * Jm * r;
                points.push_back(r);
            }
            sum *= segments[i].D;
            err = abs(sum - expct);
        }
        error += err;
        raxis->addOrderedPoints(points.begin(), points.end());
        segments[i].weights = weights_cache[j - 1];
    }

    SOLVER->writelog(LOG_DETAIL, "Sampling structure in {:d} points (error: {:g}/{:g})", raxis->size(), error / expected,
                     SOLVER->integral_error);

    // Allocate memory for integrals
    size_t nlayers = solver->lcount;
    layers_integrals.resize(nlayers);

    mesh = plask::make_shared<RectangularMesh<2>>(raxis, solver->verts, RectangularMesh<2>::ORDER_01);

    m_changed = false;
}

void ExpansionBessel::prepareIntegrals(double lam, double glam) {
    if (m_changed) init2();
    temperature = SOLVER->inTemperature(mesh);
    gain_connected = SOLVER->inGain.hasProvider();
    if (gain_connected) {
        if (isnan(glam)) glam = lam;
        gain = SOLVER->inGain(mesh, glam);
    }
}

void ExpansionBessel::cleanupIntegrals(double, double) {
    temperature.reset();
    gain.reset();
}

void ExpansionBessel::integrateParams(Integrals& integrals,
                                      const dcomplex* epsp_data, const dcomplex* iepsr_data, const dcomplex* epsz_data) {
    auto raxis = mesh->tran();
    
    size_t nr = raxis->size(), N = SOLVER->size;
    double ib = 1. / rbounds[rbounds.size() - 1];

    integrals.reset(N);

    zero_matrix(integrals.Tsp);
    zero_matrix(integrals.Tps);

    // Scale factors for making matrices orthonormal
    aligned_unique_ptr<double> factors(aligned_malloc<double>(N));
    double R = rbounds[rbounds.size() - 1];
    for (size_t i = 0; i < N; ++i) {
        double fact =R * cyl_bessel_j(m+1, kpts[i]);
        factors.get()[i] = 2. / (fact * fact);
    }

    aligned_unique_ptr<double> Jm(aligned_malloc<double>(N));
    aligned_unique_ptr<double> Jp(aligned_malloc<double>(N));

    // Compute integrals
    for (size_t ri = 0; ri != nr; ++ri) {
        double r = raxis->at(ri);
        const dcomplex iepsr = iepsr_data[ri];

        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            double kr = k * r;
            Jm.get()[i] = cyl_bessel_j(m-1, kr);
            Jp.get()[i] = cyl_bessel_j(m+1, kr);
        }

        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                dcomplex c = r * iepsr * factors.get()[i];
                integrals.Tsp(i,j) = c * Jm.get()[i] * Jm.get()[j];
                integrals.Tps(i,j) = c * Jp.get()[i] * Jp.get()[j];
            }
        }
    }

    make_unit_matrix(integrals.Tss);
    make_unit_matrix(integrals.Tpp);

    invmult(integrals.Tsp, integrals.Tss);
    invmult(integrals.Tps, integrals.Tpp);

    TempMatrix temp = getTempMatrix();
    cmatrix work(N, N, temp.data());

    cmatrix Tsp = work, Tps = integrals.Vzz;  // just to use simpler names

    zero_matrix(Tsp);
    zero_matrix(Tps);

    for (size_t ri = 0; ri != nr; ++ri) {
        double r = raxis->at(ri);
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            double kr = k * r;
            Jm.get()[i] = cyl_bessel_j(m-1, kr);
            Jp.get()[i] = cyl_bessel_j(m+1, kr);
        }
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                dcomplex c = r * factors.get()[i];
                Tsp(i,j) += c * Jp.get()[i] * Jm.get()[j];
                Tps(i,j) += c * Jm.get()[i] * Jp.get()[j];
            }
        }
    }

    mult_matrix_by_matrix(integrals.Tss, Tsp, integrals.Tsp);
    mult_matrix_by_matrix(integrals.Tpp, Tps, integrals.Tps);

    for (size_t ri = 0; ri != nr; ++ri) {
        double r = raxis->at(ri);
        dcomplex epsp = epsp_data[ri];
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            double kr = k * r;
            Jm.get()[i] = cyl_bessel_j(m-1, kr);
            Jp.get()[i] = cyl_bessel_j(m+1, kr);
        }
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                dcomplex c = r * epsp * factors.get()[i];
                integrals.Tss(i,j) += c * Jm.get()[i] * Jm.get()[j];
                integrals.Tsp(i,j) -= c * Jm.get()[i] * Jp.get()[j];
                integrals.Tpp(i,j) += c * Jp.get()[i] * Jp.get()[j];
                integrals.Tps(i,j) -= c * Jp.get()[i] * Jm.get()[j];
            }
        }
    }

    zero_matrix(work);
    double* J = Jm.get();

    for (size_t ri = 0; ri != nr; ++ri) {
        double r = raxis->at(ri);
        dcomplex epsz = epsz_data[ri];
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            double kr = k * r;
            J[i] = cyl_bessel_j(m, kr);
        }
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < N; ++i) {
                work(i,j) += r * epsz * factors.get()[i] * J[i] * J[j];
            }
        }
    }

    // make_unit_matrix(integrals.Vzz);
    zero_matrix(integrals.Vzz);
    for (int i = 0; i < N; ++i) {
        double k = kpts[i] * ib;
        integrals.Vzz(i,i) = k;
    }
    invmult(work, integrals.Vzz);
    // for (size_t i = 0; i < N; ++i) {
    //     double g = kpts[i] * ib;
    //     for (size_t j = 0; j < N; ++j) integrals.Vzz(i,j) *= g;
    // }
}

Tensor3<dcomplex> ExpansionBessel::getEps(size_t layer, size_t ri, double r, double matz, double lam, double glam) {
    Tensor3<dcomplex> eps;
    {
        OmpLockGuard<OmpNestLock> lock;  // this must be declared before `material` to guard its destruction
        auto material = SOLVER->getGeometry()->getMaterial(vec(r, matz));
        lock = material->lock();
        eps = material->NR(lam, getT(layer, ri));
        if (isnan(eps.c00) || isnan(eps.c11) || isnan(eps.c22) || isnan(eps.c01))
            throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K", material->name(),
                           lam, getT(layer, ri));
    }
    if (!is_zero(eps.c00 - eps.c11) || eps.c01 != 0.)
        throw BadInput(solver->getId(), "Lateral anisotropy not allowed for this solver");
    if (gain_connected && solver->lgained[layer]) {
        auto roles = SOLVER->getGeometry()->getRolesAt(vec(r, matz));
        if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
            Tensor2<double> g = 0.;
            double W = 0.;
            for (size_t k = 0, v = ri * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
                if (solver->stack[k] == layer) {
                    double w =
                        (k == 0 || k == mesh->vert()->size() - 1) ? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                    g += w * gain[v];
                    W += w;
                }
            }
            Tensor2<double> ni = glam * g / W * (0.25e-7 / PI);
            eps.c00.imag(ni.c00);
            eps.c11.imag(ni.c00);
            eps.c22.imag(ni.c11);
        }
    }
    eps.sqr_inplace();
    return eps;
}

std::pair<dcomplex, dcomplex> ExpansionBessel::integrateLayer(size_t layer, double lam, double glam, bool finite) {
    if (isnan(real(k0)) || isnan(imag(k0))) throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();

    auto raxis = mesh->tran();

    #if defined(OPENMP_FOUND)  // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d}/{:d} in thread {:d}", layer, solver->lcount,
                        omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d}/{:d}", layer, solver->lcount);
    #endif

    if (isnan(lam)) throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    size_t nr = raxis->size(), N = SOLVER->size;

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    double matz;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            matz = solver->verts->at(i);
            break;
        }
    }

    // For checking if the layer is uniform
    diagonals[layer] = true;

    size_t pmli = raxis->size();
    double pmlr;
    dcomplex epsp0, iepsr0, epsz0;
    if (finite) {
        if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
            size_t pmlseg = segments.size() - 1;
            pmli -= segments[pmlseg].weights.size();
            pmlr = rbounds[pmlseg];
        }
    } else {
        double T = getT(layer, mesh->tran()->size() - 1);
        Tensor3<dcomplex> eps0;
        {
            OmpLockGuard<OmpNestLock> lock;  // this must be declared before `material` to guard its destruction
            auto material = geometry->getMaterial(vec(rbounds[rbounds.size() - 1] + 0.001, matz));
            lock = material->lock();
            eps0 = material->NR(lam, T);
            if (isnan(eps0.c00) || isnan(eps0.c11) || isnan(eps0.c22) || isnan(eps0.c01))
                throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K",
                               material->name(), lam, T);
        }
        eps0.sqr_inplace();
        epsp0 = eps0.c00;
        iepsr0 = 1. / eps0.c11;
        epsz0 = eps0.c22;
        if (abs(epsp0.imag()) < SMALL) epsp0.imag(0.);
        if (abs(iepsr0.imag()) < SMALL) iepsr0.imag(0.);
        if (abs(epsz0.imag()) < SMALL) epsz0.imag(0.);
        writelog(LOG_DEBUG, "Reference refractive index for layer {} is {} / {}", layer, str(sqrt(epsp0 + 0.5 / iepsr0)),
                 str(sqrt(epsz0)));
    }

    aligned_unique_ptr<dcomplex> epsp_data(aligned_malloc<dcomplex>(nr));
    aligned_unique_ptr<dcomplex> iepsr_data(aligned_malloc<dcomplex>(nr));
    aligned_unique_ptr<dcomplex> epsz_data(aligned_malloc<dcomplex>(nr));

    // Compute integrals
    for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
        if (wi == nw) {
            nw = segments[++seg].weights.size();
            wi = 0;
        }
        double r = raxis->at(ri);
        double w = segments[seg].weights[wi] * segments[seg].D;

        Tensor3<dcomplex> eps = getEps(layer, ri, r, matz, lam, glam);
        if (ri >= pmli) {
            dcomplex f = 1. + (SOLVER->pml.factor - 1.) * pow((r - pmlr) / SOLVER->pml.size, SOLVER->pml.order);
            eps.c00 *= f;
            eps.c11 /= f;
            eps.c22 *= f;
        }
        dcomplex epsp = eps.c00, iepsr = 1. / eps.c11, epsz = eps.c22;

        if (finite) {
            if (ri == 0) {
                epsp0 = epsp;
                iepsr0 = iepsr;
                epsz0 = epsz;
            } else {
                if (!is_zero(epsp - epsp0) || !is_zero(iepsr - iepsr0) || !is_zero(epsz - epsz0) || !is_zero(eps.c11 - epsp))
                    diagonals[layer] = false;
            }
        } else {
            epsp -= epsp0;
            iepsr -= iepsr0;
            epsz -= epsz0;
            if (!is_zero(epsp) || !is_zero(iepsr) || !is_zero(epsz)) diagonals[layer] = false;
        }

        epsp *= w;
        iepsr *= w;
        epsz *= w;

        epsp_data.get()[ri] = epsp;
        iepsr_data.get()[ri] = iepsr;
        epsz_data.get()[ri] = epsz;
    }


    if (diagonals[layer]) {
        Integrals& integrals = layers_integrals[layer];
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
        zero_matrix(integrals.Vzz);
        zero_matrix(integrals.Tss);
        zero_matrix(integrals.Tsp);
        zero_matrix(integrals.Tps);
        zero_matrix(integrals.Tpp);
        if (finite) {
            double ib = 1. / rbounds[rbounds.size() - 1];
            for (size_t i = 0; i < N; ++i) {
                double k = kpts[i] * ib;
                integrals.Vzz(i,i) = k / epsz0;
                // integrals.Tss(i,i) = integrals.Tpp(i,i) = 1. / iepsr0 + epsp0;
                // integrals.Tsp(i,i) = integrals.Tps(i,i) = 1. / iepsr0 - epsp0;
                integrals.Tss(i,i) = integrals.Tpp(i,i) = 2. * epsp0;
                integrals.Tsp(i,i) = integrals.Tps(i,i) = 0.;
            }
        }
    } else {
        integrateParams(layers_integrals[layer], epsp_data.get(), iepsr_data.get(), epsz_data.get());
    }

    return std::make_pair(2. * epsp0, 1. / epsz0);

}

// #ifndef NDEBUG
// cmatrix ExpansionBessel::epsVmm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Vmm(i,j);
//     return result;
// }
// cmatrix ExpansionBessel::epsVpp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Vpp(i,j);
//     return result;
// }
// cmatrix ExpansionBessel::epsTmm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tmm(i,j);
//     return result;
// }
// cmatrix ExpansionBessel::epsTpp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tpp(i,j);
//     return result;
// }
// cmatrix ExpansionBessel::epsTmp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tmp(i,j);
//     return result;
// }
// cmatrix ExpansionBessel::epsTpm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Tpm(i,j);
//     return result;
// }
// cmatrix ExpansionBessel::epsDm(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Dm(i,j);
//     return result;
// }
// cmatrix ExpansionBessel::epsDp(size_t layer) {
//     size_t N = SOLVER->size;
//     cmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].Dp(i,j);
//     return result;
// }
// dmatrix ExpansionBessel::epsVV(size_t layer) {
//     size_t N = SOLVER->size;
//     dmatrix result(N, N, 0.);
//     for (size_t i = 0; i != N; ++i)
//         for (size_t j = 0; j != N; ++j)
//             result(i,j) = layers_integrals[layer].VV(i,j);
//     return result;
// }
// #endif

void ExpansionBessel::prepareField() {
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_NEAREST;
}

void ExpansionBessel::cleanupField() {}

LazyData<Vec<3, dcomplex>> ExpansionBessel::getField(size_t layer,
                                                     const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                     const cvector& E,
                                                     const cvector& H) {
    size_t N = SOLVER->size;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double ib = 1. / rbounds[rbounds.size() - 1];
    const dcomplex fz = I / k0;

    auto src_mesh =
        plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    
    
    if (which_field == FIELD_E) {
        cvector Ez(N);
        {
            cvector Dz(N);
            for (size_t j = 0; j != N; ++j) {
                size_t js = idxs(j), jp = idxp(j);
                Dz[j] = fz * (H[js] - H[jp]);
            }
            mult_matrix_by_vector(layers_integrals[layer].Vzz, Dz, Ez);
        }
        return LazyData<Vec<3, dcomplex>>(dest_mesh->size(), [=](size_t i) -> Vec<3, dcomplex> {
            double r = dest_mesh->at(i)[0];
            Vec<3, dcomplex> result{0., 0., 0.};
            for (size_t j = 0; j != N; ++j) {
                double k = kpts[j] * ib;
                double kr = k * r;
                double Jm = cyl_bessel_j(m-1, kr), Jp = cyl_bessel_j(m+1, kr), J = cyl_bessel_j(m, kr);
                size_t js = idxs(j), jp = idxp(j);
                result.c0 += Jm * E[js] - Jp * E[jp];       // E_p
                result.c1 += Jm * E[js] + Jp * E[jp];       // E_r
                result.c2 += J * Ez;                        // E_z
            }
            return result;
        });
    } else {  // which_field == FIELD_H
        double r0 = (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) ? rbounds[rbounds.size() - 1] : INFINITY;
        return LazyData<Vec<3, dcomplex>>(dest_mesh->size(), [=](size_t i) -> Vec<3, dcomplex> {
            double r = dest_mesh->at(i)[0];
            dcomplex imu = 1.;
            if (r > r0) imu = 1. / (1. + (SOLVER->pml.factor - 1.) * pow((r - r0) / SOLVER->pml.size, SOLVER->pml.order));
            Vec<3, dcomplex> result{0., 0., 0.};
            for (size_t j = 0; j != N; ++j) {
                double k = kpts[j] * ib;
                double kr = k * r;
                double Jm = cyl_bessel_j(m-1, kr), Jp = cyl_bessel_j(m+1, kr), J = cyl_bessel_j(m, kr);
                size_t js = idxs(j), jp = idxp(j);
                result.c0 += Jm * H[js] - Jp * H[jp];               // H_p
                result.c1 += Jm * H[js] + Jp * H[jp];               // H_r
                result.c2 -= fz * k * imu * J * (E[js] - E[jp]);    // H_z
            }
            return result;
        });
    }
}

LazyData<Tensor3<dcomplex>> ExpansionBessel::getMaterialNR(size_t layer,
                                                           const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                           InterpolationMethod interp) {
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_NEAREST;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    double lam, glam;
    if (!isnan(lam0)) {
        lam = lam0;
        glam = (solver->always_recompute_gain) ? real(02e3 * PI / k0) : lam;
    } else {
        lam = glam = real(02e3 * PI / k0);
    }

    auto raxis = mesh->tran();

    DataVector<Tensor3<dcomplex>> nrs(raxis->size());
    for (size_t i = 0; i != nrs.size(); ++i) {
        Tensor3<dcomplex> eps = getEps(layer, i, raxis->at(i), level->vpos(), lam, glam);
        nrs[i] = eps.sqrt();
    }

    auto src_mesh =
        plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    return interpolate(
        src_mesh, nrs, dest_mesh, interp,
        InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::POSITIVE, InterpolationFlags::Symmetry::NO));
}

}}}  // namespace plask::optical::slab
