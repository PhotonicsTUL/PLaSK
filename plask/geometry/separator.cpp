#include "separator.h"

namespace plask {

template < int dim >
GeometryObject::Type GeometryObjectSeparator<dim>::getType() const { return GeometryObject::TYPE_SEPARATOR; }

template < int dim >
std::string GeometryObjectSeparator<dim>::getTypeName() const { return NAME; }

template < int dim >
shared_ptr<Material> GeometryObjectSeparator<dim>::getMaterial(const GeometryObjectSeparator<dim>::DVec &p) const {
    return shared_ptr<Material>();
}

/*template < int dim >
void GeometryObjectSeparator<dim>::getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryObject>, GeometryObjectSeparator<dim>::Box, GeometryObjectSeparator<dim>::DVec> > &dest, const PathHints *path) const {
    // do nothing
}*/

template < int dim >
void GeometryObjectSeparator<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate &predicate, std::vector<GeometryObjectSeparator<dim>::Box> &dest, const PathHints *path) const {
    //do nothing
    //if (predicate(*this)) dest.push_back(this->getBoundingBox());
}

template < int dim >
void GeometryObjectSeparator<dim>::getObjectsToVec(const GeometryObject::Predicate &predicate, std::vector<shared_ptr<const GeometryObject> > &dest, const PathHints *path) const {
    if (predicate(*this)) dest.push_back(this->shared_from_this());
}

template < int dim >
void GeometryObjectSeparator<dim>::getPositionsToVec(const GeometryObject::Predicate &predicate, std::vector<GeometryObjectSeparator<dim>::DVec> &dest, const PathHints *) const {
    if (predicate(*this)) dest.push_back(Primitive<dim>::ZERO_VEC);
}

template < int dim >
bool GeometryObjectSeparator<dim>::hasInSubtree(const GeometryObject &el) const {
    return &el == this;
}

template < int dim >
GeometryObject::Subtree GeometryObjectSeparator<dim>::getPathsTo(const GeometryObject &el, const PathHints *path) const {
    return GeometryObject::Subtree( &el == this ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
}

template < int dim >
GeometryObject::Subtree GeometryObjectSeparator<dim>::getPathsAt(const GeometryObjectSeparator<dim>::DVec &point, bool) const {
    return GeometryObject::Subtree( this->contains(point) ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
}

template < int dim >
shared_ptr<GeometryObject> GeometryObjectSeparator<dim>::getChildNo(std::size_t child_no) const {
    throw OutOfBoundsException("GeometryObjectLeaf::getChildNo", "child_no");
}

template < int dim >
shared_ptr<const GeometryObject> GeometryObjectSeparator<dim>::changedVersion(const GeometryObject::Changer &changer, Vec<3, double> *translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    changer.apply(result, translation);
    return result;
}

template < int dim >
bool GeometryObjectSeparator<dim>::contains(const GeometryObjectSeparator<dim>::DVec &p) const {
    return false;
}





template struct GeometryObjectSeparator<2>;
template struct GeometryObjectSeparator<3>;



}
