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

namespace plask { namespace solvers { namespace slab {

ExpansionBessel::ExpansionBessel(BesselSolverCyl* solver): Expansion(solver), initialized(false)
{
}


size_t ExpansionBessel::lcount() const {
    return SOLVER->getLayersPoints().size();
}


size_t ExpansionBessel::matrixSize() const {
    return 2 * SOLVER->size; // TODO should be N for m = 0?
} 


void ExpansionBessel::computeBesselZeros()
{
    unsigned m = SOLVER->m;
    size_t N = SOLVER->size;
    size_t n = 0;
    factors.resize(N);
    if (m < 5) {
        n = min(N, size_t(100));
        std::copy_n(bessel_zeros[m], n, factors.begin());
    }
    if (n < N) {
        SOLVER->writelog(LOG_DEBUG, "Computing Bessel function J_(%d) zeros %d to %d", m, n+1, N);
        cyl_bessel_j_zero(double(m), n+1, N-n, factors.begin()+n);
    }
    // #ifndef NDEBUG
    //     for (size_t i = 0; i != N; ++i) { 
    //         auto report = [i,m,this]()->bool{
    //             std::cerr << "J(" << m << ", " << factors[i] << ") = " << cyl_bessel_j(m, factors[i]) << "\n";
    //             return false;
    //         };
    //         assert(is_zero(cyl_bessel_j(m, factors[i]), 1e-9) || report());
    //     }
    // #endif
}

void ExpansionBessel::init()
{
    // Initialize segments
    if (!SOLVER->mesh) {
        SOLVER->writelog(LOG_INFO, "Creating simple mesh");
        SOLVER->setMesh(make_shared<OrderedMesh1DSimpleGenerator>(true));
    }
    rbounds = OrderedAxis(*SOLVER->getMesh());
    size_t nseg = rbounds.size() - 1;
    if (SOLVER->pml.dist > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.dist);
    if (SOLVER->pml.size > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.size);
    segments.resize(nseg);

    computeBesselZeros();

    // Estimate necessary number of integration points
    unsigned m = SOLVER->m;
    double k = factors[factors.size()-1];

    double expected = cyl_bessel_j(m+1, k) * rbounds[rbounds.size()-1];
    expected = 0.5 * expected*expected;

    k /= rbounds[rbounds.size()-1];

    double max_error = SOLVER->integral_error * expected / nseg;
    double error = 0.;
    
    std::deque<std::vector<double>> abscissae_cache;
    std::deque<DataVector<double>> weights_cache;
    
    raxis = make_shared<OrderedAxis>();
    
    double a, b = 0.;
    double expcts = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        a = b; b = rbounds[i+1];
        segments[i].Z = 0.5 * (a + b);
        segments[i].D = 0.5 * (b - a);
        
        // excpected value is the second Lommel's integral
        double expct = expcts;
        expcts = cyl_bessel_j(m, k*b); expcts = 0.5 * b*b * (expcts*expcts - cyl_bessel_j(m-1, k*b) * cyl_bessel_j(m+1, k*b));
        expct = expcts - expct;
        
        double err = 2 * max_error;
        std::vector<double> points;
        size_t j, n = 0;
        double sum;
        for (j = 0; err > max_error && n <= SOLVER->max_itegration_points; ++j) {
            n = 4 * (j+1) - 1;
            if (j == abscissae_cache.size()) {
                abscissae_cache.push_back(std::vector<double>());
                weights_cache.push_back(DataVector<double>());
                gaussData(n, abscissae_cache.back(), weights_cache.back());
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

    SOLVER->writelog(LOG_DETAIL, "Sampling structure in %d points (error: %g/%g)", raxis->size(), error/expected, SOLVER->integral_error);
    
    // Allocate memory for integrals
    size_t nlayers = lcount();
    layers_integrals.resize(nlayers);
    iepsilons.resize(nlayers);
    for (size_t l = 0, nr = raxis->size(); l != nlayers; ++l)
        iepsilons[l].reset(nr);
    diagonals.assign(nlayers, false);
    
    initialized = true;
}


void ExpansionBessel::reset()
{
    segments.clear();
    layers_integrals.clear();
    iepsilons.clear();
    factors.clear();
    initialized = false;
    raxis.reset();
}


void ExpansionBessel::layerIntegrals(size_t layer, double lam, double glam)
{
    if (isnan(real(SOLVER->k0)) || isnan(imag(SOLVER->k0)))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();
    auto zaxis = SOLVER->getLayerPoints(layer);

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer %d in thread %d", layer, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer %d", layer);
    #endif

    size_t nr = raxis->size(), N = SOLVER->size;
    double ib = 1. / rbounds[rbounds.size()-1];
    int m = int(SOLVER->m);

    auto mesh = make_shared<RectangularMesh<2>>(raxis, zaxis, RectangularMesh<2>::ORDER_01);

    LazyData<double> gain;
    auto temperature = SOLVER->inTemperature(mesh);
    bool gain_connected = SOLVER->inGain.hasProvider(), gain_computed = false;

    double matz = zaxis->at(0); // at each point along any vertical axis material is the same

    Integrals& integrals = layers_integrals[layer];
    integrals.reset(N);
    
    // For checking if the layer is uniform
    dcomplex eps0;
    diagonals[layer] = true;
    
    // Compute integrals
    for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
        if (wi == nw) {
            nw = segments[++seg].weights.size();
            wi = 0;
        }
        double r = raxis->at(ri);
        double w = segments[seg].weights[wi] * segments[seg].D;

        auto material = geometry->getMaterial(vec(r, matz));
        double T = 0.; for (size_t v = ri * zaxis->size(), end = (ri+1) * zaxis->size(); v != end; ++v) T += temperature[v]; T /= zaxis->size();
        dcomplex eps = material->Nr(lam, T);
        if (gain_connected &&  SOLVER->lgained[layer]) {
            auto roles = geometry->getRolesAt(vec(r, matz));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                if (!gain_computed) {
                    gain = SOLVER->inGain(mesh, glam);
                    gain_computed = true;
                }
                double g = 0.; for (size_t v = ri * zaxis->size(), end = (ri+1) * zaxis->size(); v != end; ++v) g += gain[v];
                double ni = glam * g/zaxis->size() * (0.25e-7/M_PI);
                eps.imag(ni);
            }
        }
        eps = eps * eps;
        dcomplex ieps = 1. / eps;
        iepsilons[layer][ri] = ieps;
        
        if (ri == 0) eps0 = eps;
        else if (!is_zero(eps - eps0)) diagonals[layer] = false;
        
        ieps *= w;
        eps *= w;
        
        for (int i = 0; i < N; ++i) {
            double g = factors[i] * ib; double gr = g*r;
            double Jmg = cyl_bessel_j(m-1, gr), Jpg = cyl_bessel_j(m+1, gr), Jg = cyl_bessel_j(m, gr),
                   Jm2g = cyl_bessel_j(m-2, gr), Jp2g = cyl_bessel_j(m+2, gr);
                    
            for (int j = i; j < N; ++j) {
                double k = factors[j] * ib; double kr = k*r;
                double Jmk = cyl_bessel_j(m-1, kr), Jpk = cyl_bessel_j(m+1, kr), Jk = cyl_bessel_j(m, kr);
                
                integrals.ieps_minus(i,j) += r * Jmg * ieps * Jmk;
                integrals.ieps_plus(i,j)  += r * Jpg * ieps * Jpk;
                integrals.eps_minus(i,j)  += r * Jmg * eps * Jmk;
                integrals.eps_plus(i,j)   += r * Jpg * eps * Jpk;
                
                integrals.deps_minus(i,j) -= ieps * (0.5*r*(g*(Jm2g-Jg)*Jk + k*Jmg*(Jmk-Jpk)) + Jmg*Jk);
                integrals.deps_plus(i,j)  -= ieps * (0.5*r*(g*(Jg-Jp2g)*Jk + k*Jpg*(Jmk-Jpk)) + Jpg*Jk);

                if (j != i) {
                    double Jm2k = cyl_bessel_j(m-2, kr), Jp2k = cyl_bessel_j(m+2, kr);
                    integrals.deps_minus(j,i) -= ieps * (0.5*r*(k*(Jm2k-Jk)*Jg + g*Jmk*(Jmg-Jpg)) + Jmk*Jg);
                    integrals.deps_plus(j,i)  -= ieps * (0.5*r*(k*(Jk-Jp2k)*Jg + g*Jpk*(Jmg-Jpg)) + Jpk*Jg);
                }
            }
        }
    }
    
    if (diagonals[layer]) {
        SOLVER->writelog(LOG_DETAIL, "Layer %1% is uniform", layer);
        integrals.zero();
        for (int i = 0; i < N; ++i) {
            double val = cyl_bessel_j(m+1, factors[i]) * rbounds[rbounds.size()-1]; val = 0.5 * val*val;;
            integrals.ieps_minus(i,i) = integrals.ieps_plus(i,i) = val / eps0;
            integrals.eps_minus(i,i) = integrals.eps_plus(i,i) = val * eps0;
        }
    }
}


#ifndef NDEBUG
cmatrix ExpansionBessel::ieps_minus(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].ieps_minus(i,j);
    return result;
}

cmatrix ExpansionBessel::ieps_plus(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].ieps_plus(i,j);
    return result;
}

cmatrix ExpansionBessel::eps_minus(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].eps_minus(i,j);
    return result;
}

cmatrix ExpansionBessel::eps_plus(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].eps_plus(i,j);
    return result;
}

cmatrix ExpansionBessel::deps_minus(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].deps_minus(i,j);
    return result;
}

cmatrix ExpansionBessel::deps_plus(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].deps_plus(i,j);
    return result;
}

#endif
                                              


void ExpansionBessel::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH)
{
    size_t N = SOLVER->size;
    int m = int(SOLVER->m);
    dcomplex k0 = SOLVER->k0;
    dcomplex f0 = 1. / k0;
    double b = rbounds[rbounds.size()-1];
    
    Integrals& braket = layers_integrals[layer];
    
    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double f = 1. / (cyl_bessel_j(m+1, factors[i]) * b); f *= f;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            double k = factors[j] / b;
            RH(is, js) = 0.;
            RH(ip, js) = 0.;
            RH(is, jp) = - f * f0 * k * (k * (braket.ieps_minus(i,j) - braket.ieps_plus(i,j)) + (braket.deps_minus(i,j) + braket.deps_plus(i,j)));
            RH(ip, jp) = - f * f0 * k * (k * (braket.ieps_minus(i,j) + braket.ieps_plus(i,j)) + (braket.deps_minus(i,j) - braket.deps_plus(i,j)));
        }
        RH(is, is) += k0;
        RH(ip, ip) += k0;
    }

    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double f = 1. / (cyl_bessel_j(m+1, factors[i]) * b); f *= f;
        double gg = factors[i] / b; gg *= gg;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            RE(is, js) = RE(ip, jp) = f * k0 * (braket.eps_minus(i,j) + braket.eps_plus(i,j));
            RE(ip, js) = RE(is, jp) = f * k0 * (braket.eps_minus(i,j) - braket.eps_plus(i,j));
        }
        RE(is, is) -= f0 * gg;
    }
}

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
    int m = SOLVER->m;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double b = rbounds[rbounds.size()-1];

    auto src_mesh = make_shared<RectangularMesh<2>>(raxis, make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
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
                    double kr = r * factors[j] / b;
                    double Jm = cyl_bessel_j(m-1, kr),
                           Jp = cyl_bessel_j(m+1, kr),
                           J = cyl_bessel_j(m, kr);
                    double A = Jm + Jp, B = Jm - Jp;
                    result.c0 -= A * E[idxp(j)] + B * E[idxs(j)];   // E_p
                    result.c1 += A * E[idxs(j)] + B * E[idxp(j)];   // E_r
                    double k = factors[j] / b;
                    result.c2 += 2. * k * ieps[j] * J * H[idxp(j)]; // E_z
                }
                return result;
            });
    } else { // which_field == FIELD_H
        return LazyData<Vec<3,dcomplex>>(dest_mesh->size(), 
            [=](size_t i) -> Vec<3,dcomplex> {
                double r = dest_mesh->at(i)[0];
                Vec<3,dcomplex> result {0., 0., 0.};
                for (size_t j = 0; j != N; ++j) {
                    double kr = r * factors[j] / b;
                    double Jm = cyl_bessel_j(m-1, kr),
                           Jp = cyl_bessel_j(m+1, kr),
                           J = cyl_bessel_j(m, kr);
                    double A = Jm + Jp, B = Jm - Jp;
                    result.c0 += A * H[idxs(j)] + B * H[idxp(j)];   // H_p
                    result.c1 += A * H[idxp(j)] + B * H[idxs(j)];   // H_r
                    double k = factors[j] / b;
                    result.c2 += 2. * k * J * E[idxs(j)];           // H_z
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
    
    auto src_mesh = make_shared<RectangularMesh<2>>(raxis, make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    return interpolate(src_mesh, nrs, dest_mesh, interp,
                       InterpolationFlags(SOLVER->getGeometry(), 
                                          InterpolationFlags::Symmetry::POSITIVE,
                                          InterpolationFlags::Symmetry::NO));
}


}}} // # namespace plask::solvers::slab
