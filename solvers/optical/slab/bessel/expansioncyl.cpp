#include "expansioncyl.h"
#include "solvercyl.h"
#include "zeros-data.h"
#include "../patterson.h"
#include "../patterson-data.h"

#include <boost/math/special_functions/bessel.hpp>
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionBessel::ExpansionBessel(BesselSolverCyl* solver): Expansion(solver), initialized(false)
{
}


size_t ExpansionBessel::lcount() const {
    return SOLVER->getLayersPoints().size();
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
}

void ExpansionBessel::init()
{
    // Initialize segments
    if (!rbounds) {
        auto mesh = RectilinearMesh2DSimpleGenerator(true)(SOLVER->geometry->getChild());
        rbounds = dynamic_pointer_cast<OrderedAxis>(static_pointer_cast<RectangularMesh<2>>(mesh)->axis0);
    }
    size_t nseg = rbounds->size() - 1;
    segments.resize(nseg);
    for (size_t i = 0; i != nseg; ++i) {
        segments[i].Z = 0.5 * (rbounds->at(i) + rbounds->at(i+1));
        segments[i].D = 0.5 * (rbounds->at(i+1) - rbounds->at(i));
    }

    computeBesselZeros();
    
    // Estimate necessary number of integration points
    unsigned m = SOLVER->m;
    double total = 0.;
    double b = rbounds->at(rbounds->size()-1);
    double a = factors[factors.size()-1] / b;
    double expected = cyl_bessel_j(m+1, a); expected = 0.5 * expected*expected;
    double err = 2. * SOLVER->integral_error;
    bool can_refine = true;
    
    //TODO maybe this loop can be more effective if done by hand...
    while (abs(1. - total/expected) > SOLVER->integral_error && can_refine) {
        err *= 0.5;
        can_refine = false;
        for (size_t i = 0; i < nseg; ++i) {
            double e = err * b / segments[i].D;
            auto fun = [m, a](double r) -> double { double j = cyl_bessel_j(m, a*r); return j*j*r; };
            total += patterson<double,double>(fun, rbounds->at(i), rbounds->at(i+1), e, &segments[i].n);
            if (segments[i].n < 8) can_refine = true;
        }
    }
    
    size_t n; for (size_t i = 0; i < nseg; ++i) n += (2 << segments[i].n) - 1;
    std::vector<double> points;
    points.reserve(n);
    
    for (size_t i = 0; i < nseg; ++i) {
        const double stp = 256 >> segments[i].n;
        for (unsigned j = 256 - stp; j > 0; j -= stp) points.push_back(segments[i].Z - segments[i].D * patterson_points[j]);
        for (unsigned j = 0; j < 256; j += stp) points.push_back(segments[i].Z + segments[i].D * patterson_points[j]);
    }
    raxis.reset(new OrderedAxis(std::move(points), 0.));
    
    // Allocate memory for integrals
    size_t nlayers = lcount();
    layers_integrals.resize(nlayers);
    diagonals.assign(nlayers, false);
    
    initialized = true;
}


void ExpansionBessel::reset()
{
    segments.clear();
    layers_integrals.clear();
    factors.clear();
    initialized = false;
    raxis.reset();
}


void ExpansionBessel::layerIntegrals(size_t layer)
{
    if (isnan(real(SOLVER->k0)) || isnan(imag(SOLVER->k0)))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();
    auto zaxis = SOLVER->getLayerPoints(layer);

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DEBUG, "Computing integrals for layer %d in thread %d", layer, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DEBUG, "Computing integrals for layer %d", l);
    #endif

    size_t nseg = rbounds->size() - 1, nr = raxis->size(), N = SOLVER->size;
    double ib = 1. / rbounds->at(rbounds->size()-1);
    int m = int(SOLVER->m);

    auto mesh = make_shared<RectangularMesh<2>>(raxis, zaxis, RectangularMesh<2>::ORDER_01);
    double lambda = real(2e3*M_PI/SOLVER->k0);

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
    ptrdiff_t wi = - 1 << segments[0].n + 1;
    for (size_t ri = 0, seg = 0, nw = 1<<segments[0].n, hw = 1<<segments[0].n; ri != nr; ++ri, ++wi) {
        if (wi == nw) {
            wi = 1;
            nw = 1 << segments[++seg].n;
            hw = - nw + 1;
        }
        double r = raxis->at(ri);
        double w = patterson_weights[segments[seg].n][abs(wi)] * segments[seg].D;

        auto material = geometry->getMaterial(vec(r, matz));
        double T = 0.; for (size_t v = ri * zaxis->size(), end = (ri+1) * zaxis->size(); v != end; ++v) T += temperature[v]; T /= zaxis->size();
        dcomplex eps = material->Nr(lambda, T);
        if (gain_connected) {
            auto roles = geometry->getRolesAt(vec(r, matz));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                if (!gain_computed) {
                    gain = SOLVER->inGain(mesh, lambda);
                    gain_computed = true;
                }
                double g = 0.; for (size_t v = ri * zaxis->size(), end = (ri+1) * zaxis->size(); v != end; ++v) g += gain[v];
                double ni = lambda * g/zaxis->size() * (0.25e-7/M_PI);
                eps.imag(ni);
            }
        }
        eps = eps * eps;
        dcomplex ieps = 1. / eps;
        
        if (ri == 0) eps0 = eps;
        else if (!is_zero(eps - eps0)) diagonals[layer] = false;
        
        for (int i = 0; i < N; ++i) {
            double g = factors[i] * ib; double gr = g*r;
            for (int j = i; j < N; ++j) {
                double k = factors[j] * ib; double kr = k*r;
                
                double Jmg = cyl_bessel_j(m-1, gr), Jpg = cyl_bessel_j(m+1, gr), Jg = cyl_bessel_j(m, kr),
                       Jm2g = cyl_bessel_j(m-2, gr), Jp2g = cyl_bessel_j(m+2, gr);
                double Jmk = cyl_bessel_j(m-1, kr), Jpk = cyl_bessel_j(m+1, kr), Jk = cyl_bessel_j(m, kr);
                
                integrals.ieps_minus(i,j) += w * r * Jmg * ieps * Jmk;
                integrals.ieps_plus(i,j)  += w * r * Jpg * ieps * Jpk;
                integrals.eps_minus(i,j)  += w * r * Jmg * eps * Jmk;
                integrals.eps_plus(i,j)   += w * r * Jpg * eps * Jpk;
                
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
        solver->writelog(LOG_DETAIL, "Layer %1% is uniform", layer);
    }
}


void ExpansionBessel::getMatrices(size_t l, cmatrix& RE, cmatrix& RH)
{
}

void ExpansionBessel::prepareField()
{
}

void ExpansionBessel::cleanupField()
{
}

DataVector<const Vec<3,dcomplex>> ExpansionBessel::getField(size_t l,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    const cvector& E, const cvector& H)
{
}

LazyData<Tensor3<dcomplex>> ExpansionBessel::getMaterialNR(size_t layer,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    InterpolationMethod interp)
{
    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    auto geometry = SOLVER->getGeometry();
    auto zaxis = SOLVER->getLayerPoints(layer);

    auto mesh = make_shared<RectangularMesh<2>>(raxis, zaxis, RectangularMesh<2>::ORDER_01);
    double lambda = real(2e3*M_PI/SOLVER->k0);

    LazyData<double> gain;
    auto temperature = SOLVER->inTemperature(mesh);
    bool gain_connected = SOLVER->inGain.hasProvider(), gain_computed = false;

    double matz = zaxis->at(0); // at each point along any vertical axis material is the same

    DataVector<Tensor3<dcomplex>> result(raxis->size());
    
    for (size_t ri = 0; ri != raxis->size(); ++ri) {
        double r = raxis->at(ri);
        auto material = geometry->getMaterial(vec(r, matz));
        double T = 0.; for (size_t v = ri * zaxis->size(), end = (ri+1) * zaxis->size(); v != end; ++v) T += temperature[v]; T /= zaxis->size();
        dcomplex nr = material->Nr(lambda, T);
        if (gain_connected) {
            auto roles = geometry->getRolesAt(vec(r, matz));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                if (!gain_computed) {
                    gain = SOLVER->inGain(mesh, lambda);
                    gain_computed = true;
                }
                double g = 0.; for (size_t v = ri * zaxis->size(), end = (ri+1) * zaxis->size(); v != end; ++v) g += gain[v];
                double ni = lambda * g/zaxis->size() * (0.25e-7/M_PI);
                nr.imag(ni);
            }
        }
        result[ri] = nr;
        auto src_mesh = make_shared<RectangularMesh<2>>(raxis, make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
        return interpolate(src_mesh, result, dest_mesh, interp,
                           InterpolationFlags(geometry, InterpolationFlags::Symmetry::POSITIVE, InterpolationFlags::Symmetry::NO)
                          );
    }
}



}}} // # namespace plask::solvers::slab
