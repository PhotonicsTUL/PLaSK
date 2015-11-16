#include "mirror.h"
#include "reader.h"
#include "../manager.h"

namespace plask {

template <int dim>
std::string Flip<dim>::getTypeName() const { return NAME; }

template <int dim>
shared_ptr<Material> Flip<dim>::getMaterial(const Flip::DVec &p) const {
    return this->hasChild() ? this->_child->getMaterial(fliped(p)) : shared_ptr<Material>();
}

template <int dim>
bool Flip<dim>::contains(const Flip<dim>::DVec &p) const {
    return this->hasChild() && this->_child->contains(fliped(p));
}

template <int dim>
GeometryObject::Subtree Flip<dim>::getPathsAt(const Flip::DVec &point, bool all) const {
    if (!this->hasChild()) return GeometryObject::Subtree();
    return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(fliped(point), all));
}

template <int dim>
typename Flip<dim>::Box Flip<dim>::fromChildCoords(const typename Flip<dim>::ChildType::Box &child_bbox) const {
    return fliped(child_bbox);
}

template <int dim>
void Flip<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    const std::size_t s = this->_child->getPositions(predicate, path).size();
    for (std::size_t i = 0; i < s; ++i) dest.push_back(Primitive<dim>::NAN_VEC);   //we can't get proper position
}

template <int dim>
shared_ptr<GeometryObjectTransform<dim> > Flip<dim>::shallowCopy() const {
    return copyShallow();
}

template <int dim>
void Flip<dim>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr("axis", axes[direction3D(flipDir)]);
}

template <int dim>
std::string Mirror<dim>::getTypeName() const { return NAME; }

template <int dim>
typename Mirror<dim>::Box Mirror<dim>::getBoundingBox() const {
    return this->hasChild() ? extended(this->_child->getBoundingBox()) : Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
}

template <int dim>
typename Mirror<dim>::Box Mirror<dim>::getRealBoundingBox() const {
    return this->hasChild() ? this->_child->getBoundingBox() : Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
}

template <int dim>
shared_ptr<Material> Mirror<dim>::getMaterial(const Mirror<dim>::DVec &p) const {
    return this->hasChild() ? this->_child->getMaterial(flipedIfNeg(p)) : shared_ptr<Material>();
}

template <int dim>
bool Mirror<dim>::contains(const Mirror<dim>::DVec &p) const {
    return this->hasChild() && this->_child->contains(flipedIfNeg(p));
}

template <int dim>
typename Mirror<dim>::Box Mirror<dim>::fromChildCoords(const typename Mirror<dim>::ChildType::Box &child_bbox) const {
    return child_bbox.extension(child_bbox.fliped(flipDir));
}

template <int dim>
void Mirror<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    if (!this->hasChild()) return;
    std::size_t old_size = dest.size();
    this->_child->getBoundingBoxesToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (std::size_t i = old_size; i < new_size; ++i)
        dest.push_back(dest[i].fliped(flipDir));
}

template <int dim>
void Mirror<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    if (!this->hasChild()) return;
    std::size_t old_size = dest.size();
    this->_child->getObjectsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (std::size_t i = old_size; i < new_size; ++i)
        dest.push_back(dest[i]);
}

template <int dim>
void Mirror<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    std::size_t old_size = dest.size();
    this->_child->getPositionsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (std::size_t i = old_size; i < new_size; ++i)
        dest.push_back(Primitive<dim>::NAN_VEC);    //we can't get proper position for fliped child
}

template <int dim>
GeometryObject::Subtree Mirror<dim>::getPathsTo(const GeometryObject& el, const PathHints* path) const {
    GeometryObject::Subtree result = GeometryObjectTransform<dim>::getPathsTo(el, path);
    if (!result.empty() && !result.children.empty())    //result.children[0] == getChild()
        result.children.push_back(GeometryObject::Subtree(plask::make_shared<Flip<dim>>(flipDir, getChild()),
                                                          result.children[0].children));
    return result;
}

template <int dim>
GeometryObject::Subtree Mirror<dim>::getPathsAt(const Mirror<dim>::DVec &point, bool all) const {
    if (!this->hasChild()) GeometryObject::Subtree();
    return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(flipedIfNeg(point), all));
}

template <int dim>
std::size_t Mirror<dim>::getChildrenCount() const {
    return this->hasChild() ? 2 : 0;
}

template <int dim>
shared_ptr<GeometryObject> Mirror<dim>::getChildNo(std::size_t child_no) const {
    if (child_no >= getChildrenCount()) throw OutOfBoundsException("getChildNo", "child_no", child_no, 0, getChildrenCount()-1);
    //child_no is 0 or 1 now, and hasChild() is true
    if (child_no == 0)
        return this->_child;
    else
        return plask::make_shared<Flip<dim>>(flipDir, this->_child);
}

template <int dim>
std::size_t Mirror<dim>::getRealChildrenCount() const {
    return GeometryObjectTransform<dim>::getChildrenCount();
}

template <int dim>
shared_ptr<GeometryObject> Mirror<dim>::getRealChildNo(std::size_t child_no) const {
    return GeometryObjectTransform<dim>::getChildNo(child_no);
}

template <int dim>
shared_ptr<GeometryObjectTransform<dim> > Mirror<dim>::shallowCopy() const {
    return copyShallow();
}

template <int dim>
void Mirror<dim>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr("axis", axes[direction3D(flipDir)]);
}


//--------- XML reading: Flip and Mirror ----------------

template <typename GeometryType>
shared_ptr<GeometryObject> read_flip_like(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, GeometryType::DIM == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    auto flipDir = reader.getAxisNames().get<GeometryType::DIM>(reader.source.requireAttribute("axis"));
    return plask::make_shared< GeometryType >(flipDir, reader.readExactlyOneChild<typename GeometryType::ChildType>(!reader.manager.draft));
}

static GeometryReader::RegisterObjectReader flip2D_reader(Flip<2>::NAME, read_flip_like<Flip<2>>);
static GeometryReader::RegisterObjectReader flip3D_reader(Flip<3>::NAME, read_flip_like<Flip<3>>);
static GeometryReader::RegisterObjectReader mirror2D_reader(Mirror<2>::NAME, read_flip_like<Mirror<2>>);
static GeometryReader::RegisterObjectReader mirror3D_reader(Mirror<3>::NAME, read_flip_like<Mirror<3>>);

template struct PLASK_API Flip<2>;
template struct PLASK_API Flip<3>;

template struct PLASK_API Mirror<2>;
template struct PLASK_API Mirror<3>;

}
