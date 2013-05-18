#include "transform_space_cylindric.h"
#include "reader.h"

namespace plask {

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
    return Box3D(
            vec(-r.upper.tran(), -r.upper.tran(), r.lower.vert()),
            vec(r.upper.tran(),  r.upper.tran(),  r.upper.vert())
           );
}

shared_ptr<GeometryObject> read_revolution(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    return make_shared<Revolution>(reader.readExactlyOneChild<typename Revolution::ChildType>());
}

static GeometryReader::RegisterObjectReader revolution_reader(Revolution::NAME, read_revolution);

}   // namespace plask
