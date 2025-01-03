/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "expansioncyl.hpp"
#include "solvercyl.hpp"
#include "zeros-data.hpp"

#include "expansioncyl-fini.hpp"
#include "expansioncyl-infini.hpp"

#include "../gauss_legendre.hpp"

#include "besselj.hpp"

#include <boost/math/special_functions/legendre.hpp>
using boost::math::legendre_p;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace modal {

ExpansionBessel::ExpansionBessel(BesselSolverCyl* solver) : Expansion(solver), m(1), initialized(false), m_changed(true) {}

size_t ExpansionBessel::matrixSize() const { return 2 * SOLVER->size; }

void ExpansionBessel::init1() {
    // Initialize segments
    if (SOLVER->mesh)
        rbounds = OrderedAxis(*SOLVER->getMesh());
    else
        rbounds = std::move(*makeGeometryGrid1D(SOLVER->getGeometry()));
    OrderedAxis::WarningOff nowarn_rbounds(rbounds);
    rbounds.addPoint(0.);
    size_t nseg = rbounds.size() - 1;
    if (dynamic_cast<ExpansionBesselFini*>(this)) {
        if (SOLVER->pml.dist > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.dist);
        if (SOLVER->pml.size > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.size);
    }
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

    double expected = rbounds[rbounds.size() - 1] * cyl_bessel_j(m + 1, k);
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

        // expected value is the second Lommel's integral
        double expct = expcts;
        expcts = cyl_bessel_j(m, k * b);
        expcts = 0.5 * b * b * (expcts * expcts - cyl_bessel_j(m - 1, k * b) * cyl_bessel_j(m + 1, k * b));
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
                double Jm = cyl_bessel_j(m, k * r);
                sum += weights_cache[j][a] * r * Jm * Jm;
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

void ExpansionBessel::beforeLayersIntegrals(dcomplex lam, dcomplex glam) {
    if (m_changed) init2();
    SOLVER->prepareExpansionIntegrals(this, mesh, lam, glam);
}

void ExpansionBessel::beforeGetEpsilon() {
    double lambda = real(2e3 * PI / k0);
    double lam, glam;
    if (!isnan(lam0)) {
        lam = lam0;
        glam = (solver->always_recompute_gain) ? lambda : lam;
    } else {
        lam = glam = lambda;
    }
    beforeLayersIntegrals(lam, glam);
}

void ExpansionBessel::afterGetEpsilon() { afterLayersIntegrals(); }

Tensor3<dcomplex> ExpansionBessel::getEps(size_t layer, size_t ri, double r, double matz, double lam, double glam) {
    Tensor3<dcomplex> eps;
    std::set<std::string> roles;
    if (epsilon_connected && solver->lcomputed[layer] || gain_connected && solver->lgained[layer])
        roles = SOLVER->getGeometry()->getRolesAt(vec(r, matz));
    bool computed = epsilon_connected && solver->lcomputed[layer] && roles.find("inEpsilon") != roles.end();
    if (computed) {
        eps = Zero<Tensor3<dcomplex>>();
        double W = 0.;
        for (size_t k = 0, v = ri * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
            if (solver->stack[k] == layer) {
                if (isnan(epsilons[v]))
                    throw BadInput(solver->getId(), "complex permittivity tensor got from inEpsilon is NaN at {}", mesh->at(v));
                double w = (k == 0 || k == mesh->vert()->size() - 1) ? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                eps += w * epsilons[v];
                W += w;
            }
        }
        eps /= W;
    } else {
        OmpLockGuard lock;  // this must be declared before `material` to guard its destruction
        auto material = SOLVER->getGeometry()->getMaterial(vec(r, matz));
        lock = material->lock();
        double T, cc;
        std::tie(T, cc) = getTC(layer, ri);
        eps = material->Eps(lam, T, cc);
        if (isnan(eps))
            throw BadInput(solver->getId(), "complex permittivity tensor (Eps) for {} is NaN at lam={}nm, T={}K, n={}/cm3",
                           material->name(), lam, T, cc);
    }
    if (!is_zero(eps.c00 - eps.c11) || eps.c01 != 0. || eps.c02 != 0. || eps.c10 != 0. || eps.c12 != 0. || eps.c20 != 0. ||
        eps.c21 != 0.)
        throw BadInput(solver->getId(), "lateral anisotropy not allowed for this solver");
    if (!computed && gain_connected && solver->lgained[layer]) {
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
            double n00 = sqrt(eps.c00).real(), n22 = sqrt(eps.c22).real();
            eps.c00 = eps.c11 = dcomplex(n00 * n00 - ni.c00 * ni.c00, 2 * n00 * ni.c00);
            eps.c22 = dcomplex(n22 * n22 - ni.c11 * ni.c11, 2 * n22 * ni.c11);
        }
    }
    return eps;
}

void ExpansionBessel::layerIntegrals(size_t layer, double lam, double glam) {
    if (isnan(real(k0)) || isnan(imag(k0))) throw BadInput(SOLVER->getId(), "no wavelength specified");

    auto geometry = SOLVER->getGeometry();

    auto raxis = mesh->tran();

#if defined(OPENMP_FOUND)  // && !defined(NDEBUG)
    SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d}/{:d} with {} rule in thread {:d}", layer, solver->lcount,
                     SOLVER->ruleName(), omp_get_thread_num());
#else
    SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d}/{:d} with {} rule", layer, solver->lcount, SOLVER->ruleName());
#endif

    if (isnan(lam)) throw BadInput(SOLVER->getId(), "no wavelength given: specify 'lam' or 'lam0'");

    size_t nr = raxis->size(), N = SOLVER->size;

    if (epsilon_connected && solver->lcomputed[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} takes some materials parameters from inEpsilon", layer);
        if (isnan(glam)) glam = lam;
    }
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

    bool finite = !dynamic_cast<ExpansionBesselInfini*>(this);

    size_t pmli = raxis->size();
    double pmlr;
    dcomplex epsp0, epsr0, epsz0;
    if (finite) {
        if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
            size_t pmlseg = segments.size() - 1;
            pmli -= segments[pmlseg].weights.size();
            pmlr = rbounds[pmlseg];
        }
    } else {
        Tensor3<dcomplex> eps0 = getEps(layer, mesh->tran()->size() - 1, rbounds[rbounds.size() - 1] + 0.001, matz, lam, glam);
        epsp0 = eps0.c00;
        if (SOLVER->rule != BesselSolverCyl::RULE_OLD) {
            epsr0 = (SOLVER->rule != BesselSolverCyl::RULE_DIRECT) ? 1. / eps0.c11 : eps0.c11;
            epsz0 = eps0.c22;
        } else {
            epsr0 = eps0.c11;
            epsz0 = 1. / eps0.c22;
        }
        if (abs(epsp0.imag()) < SMALL) epsp0.imag(0.);
        if (abs(epsr0.imag()) < SMALL) epsr0.imag(0.);
        if (abs(epsz0.imag()) < SMALL) epsz0.imag(0.);
        writelog(LOG_DEBUG, "Reference refractive index for layer {} is {} / {}", layer, str(sqrt(epsp0)),
                 str(sqrt((SOLVER->rule != BesselSolverCyl::RULE_OLD) ? epsz0 : (1. / epsz0))));
    }

    aligned_unique_ptr<dcomplex> epsp_data(aligned_malloc<dcomplex>(nr));
    aligned_unique_ptr<dcomplex> epsr_data(aligned_malloc<dcomplex>(nr));
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
        dcomplex epsp = eps.c00, epsr, epsz;
        if (SOLVER->rule != BesselSolverCyl::RULE_OLD) {
            epsr = (SOLVER->rule != BesselSolverCyl::RULE_DIRECT) ? 1. / eps.c11 : eps.c11;
            epsz = eps.c22;
        } else {
            epsr = eps.c11;
            epsz = 1. / eps.c22;
        }

        if (finite) {
            if (ri == 0) {
                epsp0 = epsp;
                epsr0 = epsr;
                epsz0 = epsz;
            } else {
                if (!is_zero(epsp - epsp0) || !is_zero(epsr - epsr0) || !is_zero(epsz - epsz0)) diagonals[layer] = false;
            }
        } else {
            epsp -= epsp0;
            epsr -= epsr0;
            epsz -= epsz0;
            if (!is_zero(epsp) || !is_zero(epsr) || !is_zero(epsz)) diagonals[layer] = false;
        }

        epsp_data.get()[ri] = epsp * w;
        epsr_data.get()[ri] = epsr * w;
        epsz_data.get()[ri] = epsz * w;
    }

    if (diagonals[layer]) {
        Integrals& integrals = layers_integrals[layer];
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
        integrals.reset(N);
        zero_matrix(integrals.V_k);
        zero_matrix(integrals.Tss);
        zero_matrix(integrals.Tsp);
        zero_matrix(integrals.Tps);
        zero_matrix(integrals.Tpp);
        double ib = 1. / rbounds[rbounds.size() - 1];
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            if (SOLVER->rule != BesselSolverCyl::RULE_OLD)
                integrals.V_k(i, i) = k / epsz0;
            else
                integrals.V_k(i, i) = k * epsz0;
            // integrals.Tss(i,i) = integrals.Tpp(i,i) = 1. / iepsr0 + epsp0;
            // integrals.Tsp(i,i) = integrals.Tps(i,i) = 1. / iepsr0 - epsp0;
            integrals.Tss(i, i) = integrals.Tpp(i, i) = 2. * epsp0;
        }
    } else {
        integrateParams(layers_integrals[layer], epsp_data.get(), epsr_data.get(), epsz_data.get(), epsp0, epsr0, epsz0);
    }
}

#ifndef NDEBUG
cmatrix ExpansionBessel::epsV_k(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j) result(i, j) = layers_integrals[layer].V_k(i, j);
    return result;
}
cmatrix ExpansionBessel::epsTss(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j) result(i, j) = layers_integrals[layer].Tss(i, j);
    return result;
}
cmatrix ExpansionBessel::epsTsp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j) result(i, j) = layers_integrals[layer].Tsp(i, j);
    return result;
}
cmatrix ExpansionBessel::epsTps(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j) result(i, j) = layers_integrals[layer].Tps(i, j);
    return result;
}
cmatrix ExpansionBessel::epsTpp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j) result(i, j) = layers_integrals[layer].Tpp(i, j);
    return result;
}
#endif

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
    const dcomplex fz = -I / k0;

    auto src_mesh =
        plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));

    if (which_field == FIELD_E) {
        cvector Ez(N);
        {
            cvector Dz(N);
            for (size_t j = 0; j != N; ++j) {
                size_t js = idxs(j), jp = idxp(j);
                Dz[j] = fz * (H[js] + H[jp]);
            }
            mult_matrix_by_vector(layers_integrals[layer].V_k, Dz, Ez);
        }
        return LazyData<Vec<3, dcomplex>>(dest_mesh->size(), [=](size_t i) -> Vec<3, dcomplex> {
            double r = dest_mesh->at(i)[0];
            Vec<3, dcomplex> result{0., 0., 0.};
            for (size_t j = 0; j != N; ++j) {
                double k = kpts[j] * ib;
                double kr = k * r;
                double Jm = cyl_bessel_j(m - 1, kr), Jp = cyl_bessel_j(m + 1, kr), J = cyl_bessel_j(m, kr);
                size_t js = idxs(j), jp = idxp(j);
                result.c0 += Jm * E[js] - Jp * E[jp];  // E_p
                result.c1 += Jm * E[js] + Jp * E[jp];  // E_r
                result.c2 += J * Ez[j];                // E_z
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
                double Jm = cyl_bessel_j(m - 1, kr), Jp = cyl_bessel_j(m + 1, kr), J = cyl_bessel_j(m, kr);
                size_t js = idxs(j), jp = idxp(j);
                result.c0 -= Jm * H[js] - Jp * H[jp];             // H_p
                result.c1 += Jm * H[js] + Jp * H[jp];             // H_r
                result.c2 += fz * k * imu * J * (E[js] + E[jp]);  // H_z
            }
            return result;
        });
    }
}

LazyData<Tensor3<dcomplex>> ExpansionBessel::getMaterialEps(size_t layer,
                                                            const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                            InterpolationMethod interp) {
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_NEAREST;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    double lam, glam;
    if (!isnan(lam0)) {
        lam = lam0;
        glam = (solver->always_recompute_gain) ? real(2e3 * PI / k0) : lam;
    } else {
        lam = glam = real(2e3 * PI / k0);
    }

    auto raxis = mesh->tran();

    DataVector<Tensor3<dcomplex>> eps(raxis->size());
    for (size_t i = 0; i != eps.size(); ++i) {
        Tensor3<dcomplex> eps = getEps(layer, i, raxis->at(i), level->vpos(), lam, glam);
    }

    auto src_mesh =
        plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    return interpolate(
        src_mesh, eps, dest_mesh, interp,
        InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::POSITIVE, InterpolationFlags::Symmetry::NO));
}

double ExpansionBessel::integratePoyntingVert(const cvector& E, const cvector& H) {
    double result = 0.;
    for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
        size_t is = idxs(i);
        size_t ip = idxp(i);
        result += real(-E[is] * conj(H[is]) + E[ip] * conj(H[ip])) * fieldFactor(i);
    }
    return 4e-12 * PI * result;  // µm² -> m²
}

double ExpansionBessel::integrateField(WhichField field,
                                       size_t layer,
                                       const cmatrix& TE,
                                       const cmatrix& TH,
                                       const std::function<std::pair<dcomplex, dcomplex>(size_t, size_t)>& vertical) {
    assert(TE.rows() == matrixSize());
    assert(TH.rows() == matrixSize());

    size_t M = TE.cols();
    assert(TH.cols() == M);

    size_t N = SOLVER->size;

    TempMatrix temp = getTempMatrix();
    cmatrix Fz(N, M, temp.data()), DBz(N, M, temp.data() + N * M);

    double R = rbounds[rbounds.size() - 1];
    double fz = 0.5 / real(k0 * conj(k0));

    if (field == FIELD_E) {
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m = 0; m < M; m++) {
            cvector Ez(N), Dz(N);
            for (size_t j = 0; j != N; ++j) {
                size_t js = idxs(j), jp = idxp(j);
                DBz(j, m) = TH(js, m) + TH(jp, m);
            }
        }
        mult_matrix_by_matrix(layers_integrals[layer].V_k, DBz, Fz);
    } else {
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m = 0; m < M; m++) {
            for (size_t j = 0; j != N; ++j) {
                size_t js = idxs(j), jp = idxp(j);
                DBz(j, m) = TE(js, m) + TE(jp, m);
            }
        }
        Fz = getHzMatrix(DBz, Fz);
    }

    double result = 0.;

    if (field == FIELD_E) {
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m1 = 0; m1 < M; ++m1) {
            for (openmp_size_t m2 = m1; m2 < M; ++m2) {
                dcomplex resxy = 0., resz = 0.;
                for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
                    double eta = fieldFactor(i);
                    size_t is = idxs(i);
                    size_t ip = idxp(i);
                    resxy += (TE(is, m1) * conj(TE(is, m2)) + TE(ip, m1) * conj(TE(ip, m2))) * eta;
                    resz += Fz(i, m1) * conj(Fz(i, m2)) * eta;
                }
                if (!(is_zero(resxy) && is_zero(resz))) {
                    auto vert = vertical(m1, m2);
                    double res = real(resxy * vert.first + fz * resz * vert.second);
                    if (m2 != m1) res *= 2;
#pragma omp atomic
                    result += res;
                }
            }
        }
    } else {
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m1 = 0; m1 < M; ++m1) {
            for (openmp_size_t m2 = m1; m2 < M; ++m2) {
                dcomplex resxy = 0., resz = 0.;
                for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
                    double eta = fieldFactor(i);
                    size_t is = idxs(i);
                    size_t ip = idxp(i);
                    resxy += (TH(is, m1) * conj(TH(is, m2)) + TH(ip, m1) * conj(TH(ip, m2))) * eta;
                    resz += Fz(i, m1) * conj(Fz(i, m2)) * eta;
                }
                if (!(is_zero(resxy) && is_zero(resz))) {
                    auto vert = vertical(m1, m2);
                    double res = real(resxy * vert.second + fz * resz * vert.first);
                    if (m2 != m1) res *= 2;
#pragma omp atomic
                    result += res;
                }
            }
        }
    }
    return 2 * PI * result;
}

}}}  // namespace plask::optical::modal
