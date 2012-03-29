#include "eim.hpp"

namespace plask { namespace eim {

dcomplex EffectiveIndex2dModule::computeMode(dcomplex beta)
{
    outBeta = beta;
    auto logger = logData<dcomplex,double>("wavelength", "char_val");
}


std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex beta1, dcomplex beta2, int steps)
{
    std::vector<dcomplex> modes = findModesMap(beta1, beta2, steps);
    for(auto mode = modes.begin(); mode != modes.end(); ++mode) {
        *mode = computeMode(*mode);
    }
}


std::vector<dcomplex> EffectiveIndex2dModule::findModesMap(dcomplex beta1, dcomplex beta2, int steps)
{
}


shared_ptr<const std::vector<double>> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method)
{
}




}} // namespace plask::eim
