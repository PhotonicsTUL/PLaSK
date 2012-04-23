#include "eim.h"

namespace plask { namespace eim {

EffectiveIndex2dModule::EffectiveIndex2dModule() :
    rootdigger(*this),
    maxiterations(500),                                     // maximum number of iterations
    tolx(1.0e-07),                                          // absolute tolerance on the argument
    tolf_min(1.0e-12),                                      // sufficient tolerance on the function value
    tolf_max(1.0e-10),                                      // required tolerance on the function value
    maxstep(0.1),                                           // maximum step in one iteration
    outNeff(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty),
    log_value(dataLog<dcomplex, double>("neff", "char_val")) {
    inTemperature = 300.;
    setSimpleMesh();
}

dcomplex EffectiveIndex2dModule::computeMode(dcomplex neff)
{
    dcomplex result = neff;

    outNeff = result;
    return result;
}


std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex neff1, dcomplex neff2, int steps, int nummodes)
{
    return rootdigger.searchSolutions(neff1, neff2, steps, 0, nummodes);
}


std::vector<dcomplex> EffectiveIndex2dModule::findModesMap(dcomplex neff1, dcomplex neff2, int steps)
{
}


shared_ptr<const std::vector<double>> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method)
{
}




}} // namespace plask::eim
