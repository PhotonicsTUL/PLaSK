#include "border.h"
#include <cmath>

#include <boost/algorithm/string.hpp>

namespace plask {

namespace border {

bool Strategy::canMoveOutsideBoundingBox() const {
    return false;
}

Strategy* Strategy::fromStr(const std::string& str, const MaterialsDB& materialsDB) {
    std::string lower_name = boost::to_lower_copy(str);
    if (lower_name == "null") return new Null();
    if (lower_name == "periodic") return new Periodic();
    if (lower_name == "extend") return new Extend();
    if (lower_name == "mirror") return new Mirror();
    return new SimpleMaterial(materialsDB.get(str));
}

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


void Mirror::apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>&) const {
    if (p > bbox_hi) p -= 2.0 * (p - bbox_hi); else
    if (p < bbox_lo) p += 2.0 * (bbox_lo - p);
}

bool Mirror::canMoveOutsideBoundingBox() const {
    return true;
}

Mirror* Mirror::clone() const {
    return new Mirror();
}


}   // namespace border

}   // namespace plask
