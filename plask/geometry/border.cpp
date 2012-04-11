#include "border.h"
#include <cmath>

namespace plask {

namespace border {


void SimpleMaterial::apply(double, double, double&, shared_ptr<plask::Material> &result_material) const {
    result_material = material;
}

SimpleMaterial* SimpleMaterial::clone() const {
    return new SimpleMaterial(this->material);
}


void Null::apply(double, double, double&, shared_ptr<plask::Material> &) const {
}

Null* Null::clone() const {
    return new Null();
}


void Extend::apply(double bbox_lo, double bbox_hi, double &p, shared_ptr<plask::Material>&) const {
    if (p < bbox_lo) p = bbox_lo;
    if (p > bbox_hi) p = bbox_hi;
}

Extend* Extend::clone() const {
    return new Extend();
}


void Periodic::apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>&) const {
    p = std::fmod(p-bbox_lo, bbox_hi-bbox_lo) + bbox_lo;
}

Periodic* Periodic::clone() const {
    return new Periodic();
}

}   // namespace border

}   // namespace plask
