#include "transform_space_cylindric.h"
#include "reader.h"

namespace plask {

bool Revolution::includes(const GeometryObjectD< 3 >::DVec& p) const {
    return getChild()->includes(childVec(p));
}

bool Revolution::intersects(const Box& area) const {
    return getChild()->intersects(childBox(area));
}

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

GeometryObject::Subtree Revolution::getPathsTo(const DVec& point) const {
    return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsTo(childVec(point)));
}

Box2D Revolution::childBox(const plask::Box3D& r) {
    Box2D result(childVec(r.lower), childVec(r.upper));
    result.fix();
    return result;
}

Box3D Revolution::parentBox(const ChildBox& r) {
    return Box3D(
            vec(-r.upper.tran(), -r.upper.tran(), r.lower.up()),
            vec(r.upper.tran(),  r.upper.tran(),  r.upper.up())
           );
}

shared_ptr<GeometryObject> read_revolution(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    return make_shared<Revolution>(reader.readExactlyOneChild<typename Revolution::ChildType>());
}

static GeometryReader::RegisterObjectReader revolution_reader(Revolution::NAME, read_revolution);

}   // namespace plask
