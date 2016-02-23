#include "transform_space_cylindric.h"
#include "reader.h"
#include "../manager.h"

#define PLASK_REVOLUTION_NAME "revolution"

namespace plask {

const char* Revolution::NAME = PLASK_REVOLUTION_NAME;

std::string Revolution::getTypeName() const { return NAME; }

bool Revolution::contains(const GeometryObjectD< 3 >::DVec& p) const {
    return this->hasChild() && this->_child->contains(childVec(p));
}


/*bool Revolution::intersects(const Box& area) const {
    return getChild()->intersects(childBox(area));
}*/

shared_ptr<Material> Revolution::getMaterial(const DVec& p) const {
    return this->hasChild() ? this->_child->getMaterial(childVec(p)) : shared_ptr<Material>();
}

Revolution::Box Revolution::fromChildCoords(const Revolution::ChildType::Box &child_bbox) const {
    return parentBox(child_bbox);
}

shared_ptr<GeometryObjectTransform< 3, GeometryObjectD<2> > > Revolution::shallowCopy() const {
    return plask::make_shared<Revolution>(this->_child);
}

GeometryObject::Subtree Revolution::getPathsAt(const DVec& point, bool all) const {
    if (!this->hasChild()) return GeometryObject::Subtree();
    return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(childVec(point), all));
}

void Revolution::getPositionsToVec(const GeometryObject::Predicate &predicate, std::vector<GeometryObjectTransformSpace::DVec> &dest, const PathHints *path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<3>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    auto child_pos_vec = this->_child->getPositions(predicate, path);
    for (const auto& v: child_pos_vec)
        dest.emplace_back(
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::quiet_NaN(),
           v.vert()    //only vert component is well defined
        );
}

bool Revolution::childIsClipped() const {
    return this->hasChild() && (this->_child->getBoundingBox().lower.tran() < 0);
}

// void Revolution::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<3> > >&dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<3> >(this->shared_from_this()));
//         return;
//     }
//     std::vector< shared_ptr<const GeometryObjectD<2> > > child_res = getChild()->extract(predicate, path);
//     for (shared_ptr<const GeometryObjectD<2>>& c: child_res)
//         dest.emplace_back(new Revolution(const_pointer_cast<GeometryObjectD<2>>(c)));
// }

/*Box2D Revolution::childBox(const plask::Box3D& r) {
    Box2D result(childVec(r.lower), childVec(r.upper));
    result.fix();
    return result;
}*/ //TODO bugy



Box3D Revolution::parentBox(const ChildBox& r) {
    double tran = std::max(r.upper.tran(), 0.0);
    return Box3D(
            vec(-tran, -tran, r.lower.vert()),
            vec(tran,  tran,  r.upper.vert())
           );
}

shared_ptr<GeometryObject> read_revolution(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    bool auto_clip = reader.source.getAttribute("auto-clip", false);
    return plask::make_shared<Revolution>(reader.readExactlyOneChild<typename Revolution::ChildType>(!reader.manager.draft), auto_clip);
    /*if (res->childIsClipped()) {
        writelog(LOG_WARNING, "Child of <revolution>, read from XPL line {0}, is implicitly clipped (to non-negative tran. coordinates).", line_nr);
    }*/
}

static GeometryReader::RegisterObjectReader revolution_reader(PLASK_REVOLUTION_NAME, read_revolution);

}   // namespace plask
