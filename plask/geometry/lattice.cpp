#include "lattice.h"
#include "reader.h"

namespace plask {

template <int dim>
typename ArrangeContainer<dim>::Box ArrangeContainer<dim>::getBoundingBox() const {
    Box bbox;
    if (!_child) {
        bbox.makeInvalid();
    } else {
        Box box = _child->getBoundingBox();
        for (int i = 0; i != dim; ++i) {
            if (translation[i] >= 0.) {
                bbox.lower[i] = box.lower[i];
                bbox.upper[i] = box.upper[i] + (int(repeat_count)-1) * translation[i];
            } else {
                bbox.lower[i] = box.lower[i] + (int(repeat_count)-1) * translation[i];
                bbox.upper[i] = box.upper[i];
            }
        }
    }
    return bbox;
};

template <int dim>
typename ArrangeContainer<dim>::Box ArrangeContainer<dim>::getRealBoundingBox() const {
    return getChild()->getBoundingBox();
}

template <int dim>
void ArrangeContainer<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                                        std::vector<ArrangeContainer<dim>::Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    if (repeat_count == 0 || !_child) return;
    std::size_t old_size = dest.size();
    _child->getBoundingBoxesToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i] + translation * r);
};

template <int dim>
void ArrangeContainer<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate,
                                  std::vector<shared_ptr<const GeometryObject>>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    if (repeat_count == 0 || !_child) return;
    std::size_t old_size = dest.size();
    _child->getObjectsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i]);
};

template <int dim>
void ArrangeContainer<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                    std::vector<typename ArrangeContainer<dim>::DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (repeat_count == 0 || !_child) return;
    std::size_t old_size = dest.size();
    _child->getPositionsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i] + translation * r);
};

template <int dim>
bool ArrangeContainer<dim>::contains(const ArrangeContainer<dim>::DVec& p) const {
    if (!_child) return false;
    auto lohi = bounds(p);
    for (int i = lohi.second; i >= lohi.first; --i)
        if (_child->contains(p - i * translation)) return true;
    return false;
};

template <int dim>
shared_ptr<Material> ArrangeContainer<dim>::getMaterial(const typename ArrangeContainer<dim>::DVec& p) const {
    if (!_child) return shared_ptr<Material>();
    auto lohi = bounds(p);
    for (int i = lohi.second; i >= lohi.first; --i)
        if (auto material = _child->getMaterial(p - i * translation)) return material;
    return shared_ptr<Material>();
};

template <int dim>
std::size_t ArrangeContainer<dim>::getChildrenCount() const {
    if (!_child) return 0;
    return repeat_count;
}

template <int dim>
shared_ptr<GeometryObject> ArrangeContainer<dim>::getChildNo(std::size_t child_no) const {
    if (child_no >= getChildrenCount())
        throw OutOfBoundsException("getChildNo", "child_no", child_no, 0, getChildrenCount()-1);
    return make_shared<Translation<dim>>(_child, child_no * translation);
}

template <int dim>
std::size_t ArrangeContainer<dim>::getRealChildrenCount() const {
    return GeometryObjectTransform<dim>::getChildrenCount();
}

template <int dim>
shared_ptr<GeometryObject> ArrangeContainer<dim>::getRealChildNo(std::size_t child_no) const {
    return GeometryObjectTransform<dim>::getRealChildNo(child_no);
}

template <int dim>
GeometryObject::Subtree ArrangeContainer<dim>::getPathsAt(const typename ArrangeContainer<dim>::DVec& point, bool all) const {
    if (!_child) return GeometryObject::Subtree();
    GeometryObject::Subtree result;
    auto lohi = bounds(point);
    if (all) {
        for (int i = lohi.first; i <= lohi.second; --i) {
            GeometryObject::Subtree child_path = _child->getPathsAt(point - i * translation, true);
            if (!child_path.empty())
                result.children.push_back(std::move(child_path));
        }
    } else {
        for (int i = lohi.second; i >= lohi.first; --i) {
            GeometryObject::Subtree child_path = _child->getPathsAt(point - i * translation, true);
            if (!child_path.empty()) {
                result.children.push_back(std::move(child_path));
                break;
            }
        }
    }
    if (!result.children.empty())
        result.object = this->shared_from_this();
    return result;
}

template <int dim>
shared_ptr<GeometryObjectTransform<dim>> ArrangeContainer<dim>::shallowCopy() const {
    return make_shared<ArrangeContainer<dim>>(_child, translation, repeat_count);
}

template <int dim>
typename ArrangeContainer<dim>::Box ArrangeContainer<dim>::fromChildCoords(const typename ArrangeContainer<dim>::ChildType::Box& child_bbox) const {
    return child_bbox;
}

template <>
void ArrangeContainer<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    if (translation.tran() != 0.) dest_xml_object.attr("d"+axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.) dest_xml_object.attr("d"+axes.getNameForVert(), translation.vert());
    dest_xml_object.attr("count", repeat_count);
    if (warn_overlapping) dest_xml_object.attr("warning", "false");
}

template <>
void ArrangeContainer<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    if (translation.lon() != 0.) dest_xml_object.attr("d"+axes.getNameForLong(), translation.lon());
    if (translation.tran() != 0.) dest_xml_object.attr("d"+axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.) dest_xml_object.attr("d"+axes.getNameForVert(), translation.vert());
    dest_xml_object.attr("count", repeat_count);
    if (warn_overlapping) dest_xml_object.attr("warning", "false");
}

shared_ptr<GeometryObject> read_arrange2d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    typename ArrangeContainer<2>::DVec vec;
    vec.tran() = reader.source.getAttribute("d"+reader.getAxisTranName(), 0.);
    vec.vert() = reader.source.getAttribute("d"+reader.getAxisVertName(), 0.);
    unsigned repeat = reader.source.requireAttribute<unsigned>("count");
    bool warn = reader.source.getAttribute("warning", true);
    auto child = reader.readExactlyOneChild<typename ArrangeContainer<2>::ChildType>();
    return make_shared<ArrangeContainer<2>>(child, vec, repeat, warn);
}

shared_ptr<GeometryObject> read_arrange3d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    typename ArrangeContainer<3>::DVec vec;
    vec.lon() = reader.source.getAttribute("d"+reader.getAxisLongName(), 0.);
    vec.tran() = reader.source.getAttribute("d"+reader.getAxisTranName(), 0.);
    vec.vert() = reader.source.getAttribute("d"+reader.getAxisVertName(), 0.);
    unsigned repeat = reader.source.requireAttribute<unsigned>("count");
    bool warn = reader.source.getAttribute("warning", true);
    auto child = reader.readExactlyOneChild<typename ArrangeContainer<3>::ChildType>();
    return make_shared<ArrangeContainer<3>>(child, vec, repeat, warn);
}

static GeometryReader::RegisterObjectReader arrange2d_reader(ArrangeContainer<2>::NAME, read_arrange2d);
static GeometryReader::RegisterObjectReader arrange3d_reader(ArrangeContainer<3>::NAME, read_arrange3d);

template struct PLASK_API ArrangeContainer<2>;
template struct PLASK_API ArrangeContainer<3>;
    
}
