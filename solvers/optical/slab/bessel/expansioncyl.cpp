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
    //TODO this loop can be more effective if done by hand...
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
//     for (size_t seg = 0; seg < nseg; ++seg) {
//         raxis.appendPoint(segments[seg].Z);
//         const unsigned nj = (2 << nstart) - 1;
//         for (unsigned j = 1; j < nj; ++j) {
//             double x = patterson_points[j];
//             raxis.appendPoint(segments[seg].Z - segments[seg].D * x);
//             raxis.appendPoint(segments[seg].Z + segments[seg].D * x);
//         }
//     }

    // Allocate memory for integrals
    size_t nlayers = lcount();
    integrals.resize(nlayers);
    diagonals.assign(nlayers, false);
    
    initialized = true;
}


void ExpansionBessel::reset()
{
    segments.clear();
    integrals.clear();
    factors.clear();
    initialized = false;
}


// void ExpansionBessel::getPattersonEpsilon(size_t layer, cvector eps, double left, double right, unsigned start, unsigned stop,
//                                           const shared_ptr<OrderedAxis>& zaxis)
// {
//     double D = (right - left) / 2., Z = (left + right) / 2.;
//     unsigned mask = (256>>start) - (256>>stop);
//     unsigned dj = (start==0)? 1 : 0;
//     unsigned len = (2<<stop) - (2<<start) + dj;
//     unsigned jz = len / 2;
//     std::vector<double> points(len);
//     if (start == 0) points[jz] = Z;
//     for (unsigned i = 0, j = 0; i < 256; ++i) {
//         if (i & mask) {
//             double x = patterson_points[i];
//             points[jz+j+dj] = Z + D*x;
//             points[jz-j-1] = Z - D*x;
//             ++j;
//         }
//     }
//     auto raxis = make_shared<OrderedAxis>(std::move(points));
//     auto mesh = make_shared<RectangularMesh<2>>(raxis, zaxis);
//     
// }


cvector ExpansionBessel::getEpsilon(size_t layer, const OrderedAxis& zaxis, const UnorderedAxis& raxis)
{
}

void ExpansionBessel::layerIntegrals(size_t layer)
{
    if (isnan(real(SOLVER->k0)) || isnan(imag(SOLVER->k0)))
        throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();
    const OrderedAxis& zaxis = SOLVER->getLayerPoints(layer);

    #if defined(OPENMP_FOUND) // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DEBUG, "Computing integrals for layer %d in thread %d", layer, omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DEBUG, "Computing integrals for layer %d", l);
    #endif

    size_t nseg = rbounds->size() - 1;

    // 1. Dzielimy według granic materiałów
    // 2. Każdą z podsekcji dzielimy na początkową ilość elementów (potęgi 2, w miarę proporcjonalnie do szerokości)
    // 3. Robimy początkową listę punktów, pobieramy materiały i zapamiętujemy
    // 4. Liczymy całki na każdym przedziale osobno
    // 5. Błąd szacujemy dla całości na podstawie całki JJ (wiemy, jaka powinna być)
    // 6. Teraz najprościej by było zagęścić każdy przedział, ale lepiej jakoś oszacować błąd na każdym przedziale by zagęścić ten co trzeba

//     std::vector<std::deque<dcomplex>> epsilons(nseg);   // epsilons got from the structure
//     std::vector<size_t> ns(nseg, 2*nstart+1);           // sizes for current retrieval of temperature and gain
// 
//     raxis.reserve(nseg * (2*nstart+1));
//     
//     // Compute initial integral (of order 3 i.e. in 15 points)
// 
//     auto eps = getEpsilon(layer, zaxis, raxis);
//     for (size_t seg = 0, delta = 0; seg < nseg; ++seg, delta += epsilons[seg].size()) {
// //         epsilons[seg].reserve(2*nstart+1);
// //         epsilons[seg]
//     }
    
        
// //     auto mesh = make_shared<RectangularMesh<2>>(rmesh, make_shared<OrderedAxis>(axis1), RectangularMesh<2>::ORDER_01);
// 
//     double lambda = real(2e3*M_PI/SOLVER->k0);
// 
//     auto temperature = SOLVER->inTemperature(mesh);
// 
//     LazyData<double> gain;
//     bool gain_connected = SOLVER->inGain.hasProvider(), gain_computed = false;
// 
//     double maty = axis1[0]; // at each point along any vertical axis material is the same
// 
//     // Make space for the result
//     DataVector<Tensor3<dcomplex>> work;
//     if (nN != nM) {
//         coeffs[l].reset(nN);
//         work.reset(nM, Tensor3<dcomplex>(0.));
//     } else {
//         coeffs[l].reset(nN, Tensor3<dcomplex>(0.));
//         work = coeffs[l];
//     }
// 
//     // Average material parameters
//     for (size_t i = 0; i != nM; ++i) {
//         auto material = geometry->getMaterial(vec(rmesh[j],maty));
//         double T = 0.; for (size_t v = j * axis1.size(), end = (j+1) * axis1.size(); v != end; ++v) T += temperature[v]; T /= axis1.size();
//         dcomplex nr = material->Nr(lambda, T);
//         if (gain_connected) {
//             auto roles = geometry->getRolesAt(vec(xmesh[j],maty));
//             if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
//                 if (!gain_computed) {
//                     gain = SOLVER->inGain(mesh, lambda);
//                     gain_computed = true;
//                 }
//                 double g = 0.; for (size_t v = j * axis1.size(), end = (j+1) * axis1.size(); v != end; ++v) g += gain[v];
//                 double ni = lambda * g/axis1.size() * (0.25e-7/M_PI);
//                 nr.imag(ni);
//             }
//         }
//         nr = nr*nr;
//     }
// 
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
