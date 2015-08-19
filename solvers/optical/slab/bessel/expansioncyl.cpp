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
    
    raxis.reset(new UnorderedAxis);
    for (size_t i = 0; i < nseg; ++i) {
        const unsigned nj = 1 << segments[i].n;
        for (unsigned j = 1; j < nj; ++j) {
            double x = patterson_points[j];
            raxis->appendPoint(segments[i].Z - segments[i].D * x);
            raxis->appendPoint(segments[i].Z + segments[i].D * x);
        }
    }

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

    auto mesh = make_shared<RectangularMesh<2>>(raxis, zaxis, RectangularMesh<2>::ORDER_01);

    double lambda = real(2e3*M_PI/SOLVER->k0);

    LazyData<double> gain;
    auto temperature = SOLVER->inTemperature(mesh);
    bool gain_connected = SOLVER->inGain.hasProvider(), gain_computed = false;

    double matz = zaxis->at(0); // at each point along any vertical axis material is the same

    Integrals& integrals = layers_integrals[layer];
    integrals.reset(N);
    
    // Compute integrals
    for (size_t i = 0, s = 0, j = 1, nj = 2<<segments[0].n; i != nr; ++i, ++j) {
        if (j == nj) {
            j = 1;
            nj = 2 << segments[++s].n;
        }
        double r = raxis->at(i);
        double w = patterson_weights[segments[s].n][j/2] * segments[s].D;
    }
    
//     // Check if the layer is uniform
//     diagonals[l] = true;
//     for (size_t i = 1; i != N; ++i) {
//         if (false) {  //TODO do the test here
//             diagonals[l] = false;
//             break;
//         }
// 
//     if (diagonals[l]) {
//         solver->writelog(LOG_DETAIL, "Layer %1% is uniform", l);
//     }
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

LazyData<Tensor3<dcomplex>> ExpansionBessel::getMaterialNR(size_t l,
                                    const shared_ptr<const typename LevelsAdapter::Level>& level,
                                    InterpolationMethod interp)
{
}



}}} // # namespace plask::solvers::slab
