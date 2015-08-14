#include "expansioncyl.h"
#include "solvercyl.h"

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionBessel::ExpansionBessel(BesselSolverCyl* solver): Expansion(solver), initialized(false)
{
}


size_t ExpansionBessel::lcount() const {
    return SOLVER->getLayersPoints().size();
}


void ExpansionBessel::init()
{
}


void ExpansionBessel::reset()
{
}


void ExpansionBessel::layerIntegrals(size_t l)
{
}


void ExpansionBessel::getMatrices(size_t l, dcomplex k0, dcomplex beta, dcomplex kx, cmatrix& RE, cmatrix& RH)
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
