#include "eim.hpp"

namespace plask { namespace eim {

dcomplex EffectiveIndex2dModule::computeMode(dcomplex neff)
{
    outNeff = neff;
}


std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex neff1, dcomplex neff2, int steps)
{
    std::vector<dcomplex> modes = findModesMap(neff1, neff2, steps);
    for(auto mode = modes.begin(); mode != modes.end(); ++mode) {
        *mode = computeMode(*mode);
    }
}


std::vector<dcomplex> EffectiveIndex2dModule::findModesMap(dcomplex neff1, dcomplex neff2, int steps)
{
}


shared_ptr<const std::vector<double>> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method)
{
}




}} // namespace plask::eim
