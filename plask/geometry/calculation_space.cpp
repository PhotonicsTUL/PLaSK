#include "calculation_space.h"

namespace plask {

shared_ptr<Material> Space2DCartesian::getMaterial(const Vec<2, double>& p) const {
    extrusion->getChild()->getMaterialOrAir(p);
}

}   // namespace plask
