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

ExpansionBessel::ExpansionBessel(BesselSolverCyl* solver): Expansion(solver), m(1),
                                                           initialized(false), m_changed(true)
{
}


size_t ExpansionBessel::matrixSize() const {
    return 2 * SOLVER->size; // TODO should be N for m = 0?
}


void ExpansionBessel::init1()
{
    // Initialize segments
    if (!SOLVER->mesh) {
        SOLVER->writelog(LOG_INFO, "Creating simple mesh");
        SOLVER->setMesh(plask::make_shared<OrderedMesh1DSimpleGenerator>(true));
    }
    rbounds = OrderedAxis(*SOLVER->getMesh());
    OrderedAxis::WarningOff nowarn_rbounds(rbounds);
    size_t nseg = rbounds.size() - 1;
    if (SOLVER->pml.dist > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.dist);
    if (SOLVER->pml.size > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.size);
    segments.resize(nseg);
    double a, b = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        a = b; b = rbounds[i+1];
        segments[i].Z = 0.5 * (a + b);
        segments[i].D = 0.5 * (b - a);
    }

    diagonals.assign(solver->lcount, false);
    initialized = true;
    m_changed = true;
}


void ExpansionBessel::reset()
{
    layers_integrals.clear();
    segments.clear();
    iepsilons.clear();
    kpts.clear();
    initialized = false;
    mesh.reset();
}


void ExpansionBessel::init3() {
    size_t nseg = rbounds.size() - 1;

    // Estimate necessary number of integration points
    double k = kpts[kpts.size()-1];

    double expected = cyl_bessel_j(m+1, k) * rbounds[rbounds.size()-1];
    expected = 0.5 * expected*expected;

    k /= rbounds[rbounds.size()-1];

    double max_error = SOLVER->integral_error * expected / nseg;
    double error = 0.;

    std::deque<std::vector<double>> abscissae_cache;
    std::deque<DataVector<double>> weights_cache;

    auto raxis = plask::make_shared<OrderedAxis>();
    OrderedAxis::WarningOff nowarn_raxis(raxis);

    double expcts = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        double b = rbounds[i+1];

        // excpected value is the second Lommel's integral
        double expct = expcts;
        expcts = cyl_bessel_j(m, k*b); expcts = 0.5 * b*b * (expcts*expcts - cyl_bessel_j(m-1, k*b) * cyl_bessel_j(m+1, k*b));
        expct = expcts - expct;

        double err = 2 * max_error;
        std::vector<double> points;
        size_t j, n = 0;
        double sum;
        for (j = 0; err > max_error && n <= SOLVER->max_integration_points; ++j) {
            n = 4 * (j+1) - 1;
            if (j == abscissae_cache.size()) {
                abscissae_cache.push_back(std::vector<double>());
                weights_cache.push_back(DataVector<double>());
                gaussLegendre(n, abscissae_cache.back(), weights_cache.back());
            }
            assert(j < abscissae_cache.size());
            assert(j < weights_cache.size());
            const std::vector<double>& abscissae = abscissae_cache[j];
            points.clear(); points.reserve(abscissae.size());
            sum = 0.;
            for (size_t a = 0; a != abscissae.size(); ++a) {
                double r = segments[i].Z + segments[i].D * abscissae[a];
                double Jm = cyl_bessel_j(m, k*r);
                sum += weights_cache[j][a] * Jm*Jm*r;
                points.push_back(r);
            }
            sum *= segments[i].D;
            err = abs(sum - expct);
        }
        error += err;
        raxis->addOrderedPoints(points.begin(), points.end());
        segments[i].weights = weights_cache[j-1];
    }

    SOLVER->writelog(LOG_DETAIL, "Sampling structure in {:d} points (error: {:g}/{:g})", raxis->size(), error/expected, SOLVER->integral_error);

    // Allocate memory for integrals
    size_t nlayers = solver->lcount;
    layers_integrals.resize(nlayers);
    iepsilons.resize(nlayers);
    for (size_t l = 0, nr = raxis->size(); l != nlayers; ++l)
        iepsilons[l].reset(nr);

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

std::pair<dcomplex, dcomplex> ExpansionBessel::integrateLayer(size_t layer, double lam, double glam, bool finite) {
    if (isnan(real(k0)) || isnan(imag(k0)))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();

    auto raxis = mesh->tran();

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d} in thread {:d}", layer, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d}", layer);
    #endif

    if (isnan(lam))
        throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    size_t nr = raxis->size(), N = SOLVER->size;
    double ib = 1. / rbounds[rbounds.size()-1];

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

    Integrals& integrals = layers_integrals[layer];
    integrals.reset(N);

    // For checking if the layer is uniform
    diagonals[layer] = true;

    size_t pmli = raxis->size();
    double pmlr;
    dcomplex epsa0 = 0., ieps0 = 0.;
    if (finite) {
        if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
            size_t pmlseg = segments.size()-1;
            pmli -= segments[pmlseg].weights.size();
            pmlr = rbounds[pmlseg];
        }
    } else {
        double T = getT(layer, mesh->tran()->size()-1);
        Tensor3<dcomplex> eps0 = geometry->getMaterial(vec(rbounds[rbounds.size()-1] + 0.001, matz))->NR(lam, T);
        if (isnan(eps0.c00) || isnan(eps0.c11) || isnan(eps0.c22) || isnan(eps0.c01))
            throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K",
                           geometry->getMaterial(vec(rbounds[rbounds.size()-1] + 0.001, matz))->name(), lam, T);
        eps0.sqr_inplace();
        epsa0 = 0.5 * (eps0.c00 + eps0.c11);
        ieps0 = 1. / eps0.c22;
        if (abs(epsa0.imag()) < SMALL) epsa0.imag(0.);
        if (abs(ieps0.imag()) < SMALL) ieps0.imag(0.);
        writelog(LOG_DEBUG, "Reference refractive index for layer {} is {} / {}", layer, str(sqrt(epsa0)), str(sqrt(1./ieps0)));
    }

    // Compute integrals
    for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
        if (wi == nw) {
            nw = segments[++seg].weights.size();
            wi = 0;
        }
        double r = raxis->at(ri);
        double w = segments[seg].weights[wi] * segments[seg].D;

        auto material = geometry->getMaterial(vec(r, matz));
        Tensor3<dcomplex> eps = material->NR(lam, getT(layer, ri));
        if (isnan(eps.c00) || isnan(eps.c11) || isnan(eps.c22) || isnan(eps.c01))
            throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K",
                           material->name(), lam, getT(layer, ri));
        if (eps.c01 != 0.)
            throw BadInput(solver->getId(), "Non-diagonal anisotropy not allowed for this solver");
        if (gain_connected &&  solver->lgained[layer]) {
            auto roles = geometry->getRolesAt(vec(r, matz));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                Tensor2<double> g = 0.; double W = 0.;
                for (size_t k = 0, v = ri * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
                    if (solver->stack[k] == layer) {
                        double w = (k == 0 || k == mesh->vert()->size()-1)? 1e-6 : solver->vbounds[k] - solver->vbounds[k-1];
                        g += w * gain[v]; W += w;
                    }
                }
                Tensor2<double> ni = glam * g/W * (0.25e-7/M_PI);
                eps.c00.imag(ni.c00); eps.c11.imag(ni.c00); eps.c22.imag(ni.c11);
            }
        }
        eps.sqr_inplace();
        if (ri >= pmli) {
            dcomplex f = 1. + (SOLVER->pml.factor - 1.) * pow((r-pmlr)/SOLVER->pml.size, SOLVER->pml.order);
            eps.c00 *= f;
            eps.c11 /= f;
            eps.c22 *= f;
        }
        dcomplex ieps = 1. / eps.c22;
        dcomplex epsa = 0.5 * (eps.c11 + eps.c00), deps = 0.5 * (eps.c11 - eps.c00);
        iepsilons[layer][ri] = ieps;

        if (finite) {
            if (ri == 0) {
                epsa0 = epsa;
                ieps0 = ieps;
            } else {
                if (!is_zero(epsa - epsa0) || !is_zero(ieps - ieps0) || !is_zero(deps)) diagonals[layer] = false;
            }
        } else {
            epsa -= epsa0;
            ieps -= ieps0;
            if (!is_zero(epsa) || !is_zero(ieps) || !is_zero(deps)) diagonals[layer] = false;
        }

        epsa *= w; deps *= w; ieps *= w;

        for (int i = 0; i < N; ++i) {
            double g = kpts[i] * ib; double gr = g*r;
            double Jmg = cyl_bessel_j(m-1, gr), Jpg = cyl_bessel_j(m+1, gr), Jg = cyl_bessel_j(m, gr),
                   Jm2g = cyl_bessel_j(m-2, gr), Jp2g = cyl_bessel_j(m+2, gr);
            for (int j = i; j < N; ++j) {
                double k = kpts[j] * ib; double kr = k*r;
                double Jmk = cyl_bessel_j(m-1, kr), Jpk = cyl_bessel_j(m+1, kr), Jk = cyl_bessel_j(m, kr);
                integrals.Vmm(i,j) += r * Jmg * ieps * Jmk;
                integrals.Vpp(i,j) += r * Jpg * ieps * Jpk;
                integrals.Tmm(i,j) += r * Jmg * epsa * Jmk;
                integrals.Tpp(i,j) += r * Jpg * epsa * Jpk;
                integrals.Tmp(i,j) += r * Jmg * deps * Jpk;
                integrals.Tpm(i,j) += r * Jpg * deps * Jmk;
                integrals.Dm(i,j) -= ieps * (0.5*r*(g*(Jm2g-Jg)*Jk + k*Jmg*(Jmk-Jpk)) + Jmg*Jk);
                integrals.Dp(i,j)  -= ieps * (0.5*r*(g*(Jg-Jp2g)*Jk + k*Jpg*(Jmk-Jpk)) + Jpg*Jk);
                if (j != i) {
                    double Jm2k = cyl_bessel_j(m-2, kr), Jp2k = cyl_bessel_j(m+2, kr);
                    integrals.Dm(j,i) -= ieps * (0.5*r*(k*(Jm2k-Jk)*Jg + g*Jmk*(Jmg-Jpg)) + Jmk*Jg);
                    integrals.Dp(j,i)  -= ieps * (0.5*r*(k*(Jk-Jp2k)*Jg + g*Jpk*(Jmg-Jpg)) + Jpk*Jg);
                }
            }
        }
    }

    if (diagonals[layer]) {
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
        integrals.zero();
        if (finite) {
            for (int i = 0; i < N; ++i) {
                double eta = cyl_bessel_j(m+1, kpts[i]) * rbounds[rbounds.size()-1]; eta = 0.5 * eta*eta;;
                integrals.Vmm(i,i) = integrals.Vpp(i,i) = eta * ieps0;
                integrals.Tmm(i,i) = integrals.Tpp(i,i) = eta * epsa0;
            }
        }
    }

    return std::make_pair(epsa0, ieps0);
}


#ifndef NDEBUG
cmatrix ExpansionBessel::epsVmm(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Vmm(i,j);
    return result;
}
cmatrix ExpansionBessel::epsVpp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Vpp(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTmm(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tmm(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTpp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tpp(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTmp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tmp(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTpm(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tpm(i,j);
    return result;
}
cmatrix ExpansionBessel::epsDm(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Dm(i,j);
    return result;
}
cmatrix ExpansionBessel::epsDp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Dp(i,j);
    return result;
}
#endif


void ExpansionBessel::prepareField()
{
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_NEAREST;
}

void ExpansionBessel::cleanupField()
{
}

LazyData<Vec<3,dcomplex>> ExpansionBessel::getField(size_t l,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    const cvector& E, const cvector& H)
{
    size_t N = SOLVER->size;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double ib = 1. / rbounds[rbounds.size()-1];
    const dcomplex fz = dcomplex(0,2) / k0;

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    auto ieps = interpolate(src_mesh, iepsilons[l], dest_mesh, field_interpolation,
                            InterpolationFlags(SOLVER->getGeometry(),
                                               InterpolationFlags::Symmetry::POSITIVE,
                                               InterpolationFlags::Symmetry::NO));

    if (which_field == FIELD_E) {
        return LazyData<Vec<3,dcomplex>>(dest_mesh->size(),
            [=](size_t i) -> Vec<3,dcomplex> {
                double r = dest_mesh->at(i)[0];
                Vec<3,dcomplex> result {0., 0., 0.};
                for (size_t j = 0; j != N; ++j) {
                    double k = kpts[j] * ib;
                    double kr = k * r;
                    double Jm = cyl_bessel_j(m-1, kr),
                           Jp = cyl_bessel_j(m+1, kr),
                           J = cyl_bessel_j(m, kr);
                    double A = Jm + Jp, B = Jm - Jp;
                    size_t js = idxs(j);
                    size_t jp = idxp(j);
                    result.c0 -= A * E[jp] + B * E[js];         // E_p
                    result.c1 += A * E[js] + B * E[jp];         // E_r
                    result.c2 += fz * k * ieps[i] * J * H[jp];  // E_z
                }
                return result;
            });
    } else { // which_field == FIELD_H
        double r0 = (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.)? rbounds[rbounds.size()-1] : INFINITY;
        return LazyData<Vec<3,dcomplex>>(dest_mesh->size(),
            [=](size_t i) -> Vec<3,dcomplex> {
                double r = dest_mesh->at(i)[0];
                dcomplex imu = 1.;
                if (r > r0) imu = 1. / (1. + (SOLVER->pml.factor - 1.) * pow((r-r0)/SOLVER->pml.size, SOLVER->pml.order));
                Vec<3,dcomplex> result {0., 0., 0.};
                for (size_t j = 0; j != N; ++j) {
                    double k = kpts[j] * ib;
                    double kr = k * r;
                    double Jm = cyl_bessel_j(m-1, kr),
                           Jp = cyl_bessel_j(m+1, kr),
                           J = cyl_bessel_j(m, kr);
                    double A = Jm + Jp, B = Jm - Jp;
                    size_t js = idxs(j);
                    size_t jp = idxp(j);
                    result.c0 += A * H[js] + B * H[jp];     // H_p
                    result.c1 += A * H[jp] + B * H[js];     // H_r
                    result.c2 += fz * k * imu * J * E[js];  // H_z
                }
                return result;
            });
    }

}

LazyData<Tensor3<dcomplex>> ExpansionBessel::getMaterialNR(size_t layer,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_NEAREST;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    DataVector<Tensor3<dcomplex>> nrs(iepsilons[layer].size());
    for (size_t i = 0; i != nrs.size(); ++i) {
        nrs[i] = Tensor3<dcomplex>(1. / sqrt(iepsilons[layer][i]));
    }

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    return interpolate(src_mesh, nrs, dest_mesh, interp,
                       InterpolationFlags(SOLVER->getGeometry(),
                                          InterpolationFlags::Symmetry::POSITIVE,
                                          InterpolationFlags::Symmetry::NO));
}


double ExpansionBessel::integratePoyntingVert(const cvector& E, const cvector& H)
{
    return 1.;
}


}}} // # namespace plask::optical::slab
