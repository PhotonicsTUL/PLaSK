#include "eim.h"

namespace plask { namespace modules { namespace eim {

EffectiveIndex2dModule::EffectiveIndex2dModule() :
    rootdigger(*this),
    symmetry(NO_SYMMETRY),
    tolx(1.0e-07),                                          // absolute tolerance on the argument
    tolf_min(1.0e-12),                                      // sufficient tolerance on the function value
    tolf_max(1.0e-10),                                      // required tolerance on the function value
    maxstep(0.1),                                           // maximum step in one iteration
    maxiterations(500),                                     // maximum number of iterations
    log_value(dataLog<dcomplex, double>("neff", "char_val")),
    outIntensity(this, &EffectiveIndex2dModule::getLightIntenisty) {
    inTemperature = 300.;
}


dcomplex EffectiveIndex2dModule::computeMode(dcomplex neff)
{
    updateCache();

    dcomplex result = rootdigger.getSolution(neff);

    outNeff = result;
    return result;
}



std::vector<dcomplex> EffectiveIndex2dModule::findModes(dcomplex neff1, dcomplex neff2, unsigned steps, unsigned nummodes)
{
    updateCache();
    return rootdigger.searchSolutions(neff1, neff2, steps, 0, nummodes);
}



std::vector<dcomplex> EffectiveIndex2dModule::findModesMap(dcomplex neff1, dcomplex neff2, unsigned steps)
{
    updateCache();

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




void EffectiveIndex2dModule::init()
{
    // Set default mesh
    if (!mesh) setSimpleMesh();

    // Create middle-points mesh
    middle_points = RectilinearMesh2d(mesh->getMidpointsMesh());
    middle_points.c0.addPoint(mesh->c0[0] - outer_distance);
    middle_points.c0.addPoint(mesh->c0[mesh->c0.size()-1] + outer_distance);
    middle_points.c1.addPoint(mesh->c1[0] - outer_distance);
    middle_points.c1.addPoint(mesh->c1[mesh->c1.size()-1] + outer_distance);

    // Assign space for refractive indices cache
    nrCache.assign(mesh->tran().size()+1, std::vector<dcomplex>(mesh->up().size()+1));
}


void EffectiveIndex2dModule::updateCache()
{
    bool updated = beforeCalculation();

    size_t xbegin = 0;

    // Some additional checks
    if (symmetry == SYMMETRY_POSITIVE || symmetry == SYMMETRY_NEGATIVE) {
        if (geometry->isSymmetric(CalculationSpace::DIRECTION_TRAN)) {
            if (updated) // Make sure we have only positive points
                for (auto x: mesh->c0) if (x < 0.) throw BadMesh(getId(), "for symmetric geometry no horizontal points can be negative");
            xbegin = 1;
        } else {
            log(LOG_WARNING, "Symmetry reset to NO_SYMMETRY for non-symmetric geometry.");
            symmetry = NO_SYMMETRY;
        }
    }

    size_t xsize = middle_points.c0.size();
    size_t ysize = middle_points.c1.size();
    size_t txmax = mesh->c0.size() - 1;
    size_t tymax = mesh->c1.size() - 1;

    if (updated || inTemperature.changed) {
        // Either temperature or structure changed, so we need to get refractive indices
        auto temperature_ptr = inTemperature(*mesh);
        const std::vector<double>& temp = *temperature_ptr;

        for (size_t i = xbegin; i != xsize; ++i) {
            size_t tx0 = (i > 0)? i - 1 : 0;
            size_t tx1 = (i < txmax)? i : txmax;
            for (size_t j = 0; j != ysize; ++j) {
                size_t ty0 = (j > 0)? j - 1 : 0;
                size_t ty1 = (j < tymax)? j : txmax;
                double T = 0.25 * ( temp[mesh->index(tx0, ty0)] + temp[mesh->index(tx0, ty1)] +
                                    temp[mesh->index(tx1, ty0)] + temp[mesh->index(tx1, ty1)] );
                nrCache[i][j] = geometry->getMaterial(middle_points(i,j))->Nr(real(inWavelength()), T);
            }
        }

    }
}



/********* Here are the computations *********/

dcomplex EffectiveIndex2dModule::char_val(dcomplex x)
{



}



shared_ptr<const std::vector<double>> EffectiveIndex2dModule::getLightIntenisty(const Mesh<2>& dst_mesh, InterpolationMethod method)
{


}




}}} // namespace plask::modules::eim
