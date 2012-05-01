#include "eim.h"

namespace plask { namespace modules { namespace eim {

EffectiveIndex2dModule::EffectiveIndex2dModule() :
    rootdigger(*this),
    changed(true),
    tolx(1.0e-07),                                          // absolute tolerance on the argument
    tolf_min(1.0e-12),                                      // sufficient tolerance on the function value
    tolf_max(1.0e-10),                                      // required tolerance on the function value
    maxstep(0.1),                                           // maximum step in one iteration
    maxiterations(500),                                     // maximum number of iterations
    log_value(dataLog<dcomplex, double>("neff", "char_val")),
    outNeff(NAN), outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
    inTemperature = 300.;
    setSimpleMesh();
}


dcomplex EffectiveIndex2dModule::computeMode(dcomplex neff)
{
    dcomplex result = neff;

    outNeff = result;
    return result;
}



std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex neff1, dcomplex neff2, unsigned steps, unsigned nummodes)
{
    return rootdigger.searchSolutions(neff1, neff2, steps, 0, nummodes);
}



std::vector<dcomplex> EffectiveIndex2dModule::findModesMap(dcomplex neff1, dcomplex neff2, unsigned steps)
{
    double rdneff = real(neff2 - neff1);
    double rneff1 = real(neff1);
    double steps1 = steps + 1;
    std::vector<double> rpoints(steps+1);
    for (unsigned i = 0; i <= steps; ++i) {
        rpoints[i] = rneff1 + rdneff * i / steps1;
    }

    std::vector<double> ipoints(1, imag(neff1));
    if (imag(neff2) != imag(neff1)) ipoints.push_back(imag(neff2));

    return rootdigger.findMap(rpoints, ipoints);
}




void EffectiveIndex2dModule::setMesh(const RectilinearMesh2d& meshxy) {
    mesh = make_shared<RectilinearMesh2d>(meshxy.getElementMesh());

    dTran.clear();
    dTran.reserve(meshxy.tran().size()-1);
    for (auto a = meshxy.tran().begin(), b = meshxy.tran().begin()+1; b != meshxy.tran().end(); ++a, ++b)
        dTran.push_back(*b - *a);

    dUp.clear();
    dUp.reserve(meshxy.tran().size()-1);
    for (auto a = meshxy.up().begin(), b = meshxy.up().begin()+1; b != meshxy.up().end(); ++a, ++b)
        dUp.push_back(*b - *a);

    changed = true;
}



void EffectiveIndex2dModule::updateCache()
{
    if (changed) {
        // We need to resize cache vectors
        nrCache.assign(mesh->tran().size()+1, std::vector<dcomplex>(mesh->up().size()+1, 1.));
    }

    if (inTemperature.changed || changed) {
        // Either temperature or structure changed, so we need to get refractive indices
    }

    changed = false;
}



/********* Here are the computations *********/

dcomplex EffectiveIndex2dModule::char_val(dcomplex x) {


}



shared_ptr<const std::vector<double>> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method)
{


}




}}} // namespace plask::modules::eim
