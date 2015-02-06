#include "transform_space_cylindric.h"
#include "reader.h"

namespace plask {

std::string Revolution::getTypeName() const { return NAME; }

bool Revolution::contains(const GeometryObjectD< 3 >::DVec& p) const {
    return getChild()->contains(childVec(p));
}


/*bool Revolution::intersects(const Box& area) const {
    return getChild()->intersects(childBox(area));
}*/

Revolution::Box Revolution::getBoundingBox() const {
    return parentBox(getChild()->getBoundingBox());
}

shared_ptr<Material> Revolution::getMaterial(const DVec& p) const {
    return getChild()->getMaterial(childVec(p));
}

void Revolution::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<ChildBox> c = getChild()->getBoundingBoxes(predicate, path);
    std::transform(c.begin(), c.end(), std::back_inserter(dest),
                   [&](const ChildBox& r) { return parentBox(r); });
}

shared_ptr<GeometryObjectTransform< 3, GeometryObjectD<2> > > Revolution::shallowCopy() const {
    return make_shared<Revolution>(this->getChild());
}

GeometryObject::Subtree Revolution::getPathsAt(const DVec& point, bool all) const {
    return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsAt(childVec(point), all));
}

bool Revolution::childIsClipped() const {
    if (!this->getChild()) return false;
    return this->getChild()->getBoundingBox().lower.tran() < 0;
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
    bool auto_clip = reader.source.getAttribute("auto_clip", false);
    return make_shared<Revolution>(reader.readExactlyOneChild<typename Revolution::ChildType>(), auto_clip);
    /*if (res->childIsClipped()) {
        writelog(LOG_WARNING, "Child of <revolution>, read from XPL line %1%, is implicitly clipped (to non-negative tran. coordinates).", line_nr);
    }*/
}

static GeometryReader::RegisterObjectReader revolution_reader(Revolution::NAME, read_revolution);

}   // namespace plask
