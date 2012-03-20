#include "eim.hpp"

namespace plask { namespace eim {

dcomplex EffectiveIndex2dModule::findMode(dcomplex beta)
{
}


std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex beta1, dcomplex beta2, int steps)
{
    std::vector<dcomplex> modes = findMap(beta1, beta2, steps);
    for(auto mode = modes.begin(); mode != modes.end(); ++mode) {
        *mode = findMode(*mode);
    }
}


std::vector<dcomplex> EffectiveIndex2dModule::findMap(dcomplex beta1, dcomplex beta2, int steps)
{
}


shared_ptr<const std::vector<double>> EffectiveIndex2dModule::getLightIntenisty(const Mesh<space::Cartesian2d>& dst_mesh, InterpolationMethod method)
{
}




}} // namespace plask::eim
