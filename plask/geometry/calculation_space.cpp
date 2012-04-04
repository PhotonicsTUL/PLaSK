#include "calculation_space.h"

namespace plask {

shared_ptr<Material> CalculationSpaceOverExtrusion::getMaterial(const Vec<2, double>& p) const {
    extrusion->getChild()->getMaterialOrAir(p);
}

}   // namespace plask
