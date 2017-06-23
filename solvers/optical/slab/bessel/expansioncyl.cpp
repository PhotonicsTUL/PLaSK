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
    segments.clear();
    iepsilons.clear();
    factors.clear();
    initialized = false;
    mesh.reset();
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

void ExpansionBessel::cleanupIntegrals(double lam, double glam) {
    temperature.reset();
    gain.reset();
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

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double b = rbounds[rbounds.size()-1];
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
                    double kr = r * factors[j] / b;
                    double Jm = cyl_bessel_j(m-1, kr),
                           Jp = cyl_bessel_j(m+1, kr),
                           J = cyl_bessel_j(m, kr);
                    double A = Jm + Jp, B = Jm - Jp;
                    result.c0 -= A * E[idxp(j)] + B * E[idxs(j)];   // E_p
                    result.c1 += A * E[idxs(j)] + B * E[idxp(j)];   // E_r
                    double k = factors[j] / b;
                    result.c2 += fz * k * ieps[i] * J * H[idxp(j)]; // E_z
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
                    double kr = r * factors[j] / b;
                    double Jm = cyl_bessel_j(m-1, kr),
                           Jp = cyl_bessel_j(m+1, kr),
                           J = cyl_bessel_j(m, kr);
                    double A = Jm + Jp, B = Jm - Jp;
                    result.c0 += A * H[idxs(j)] + B * H[idxp(j)];   // H_p
                    result.c1 += A * H[idxp(j)] + B * H[idxs(j)];   // H_r
                    double k = factors[j] / b;
                    result.c2 += fz * k * imu * J * E[idxs(j)];           // H_z
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


}}} // # namespace plask::solvers::slab
