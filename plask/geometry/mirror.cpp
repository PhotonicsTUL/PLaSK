#include "mirror.h"

namespace plask {

template <int dim>
void MirrorReflection<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<Box> result = getChild()->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    for (Box& r: result) dest.push_back(fliped(r));
}

template <int dim>
void MirrorReflection<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    const std::size_t s = getChild()->getPositions(predicate, path).size();
    for (std::size_t i = 0; i < s; ++i) dest.push_back(Primitive<dim>::NAN_VEC);   //we can't get proper position

    /*if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    const std::size_t old_size = dest.size();
    getChild()->getPositionsToVec(predicate, dest, path);
    for (std::size_t i = old_size; i < dest.size(); ++i)
        dest[i] = fliped(dest[i]);*/
}

template <int dim>
shared_ptr<const GeometryObject> MirrorReflection<dim>::changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    if (changer.apply(result, translation) || !this->hasChild()) return result;
    //TODO impl.
 /*   Vec<3, double> returned_translation(0.0, 0.0, 0.0);
    shared_ptr<const GeometryObject> new_child = this->getChild()->changedVersion(changer, &returned_translation);
    Vec<dim, double> translation_we_will_do = vec<dim, double>(returned_translation);
    if (new_child == getChild() && translation_we_will_do == Primitive<dim>::ZERO_VEC) return result;
    if (translation)    //we will change translation (partially if dim==2) internaly, so we recommend no extra translation
        *translation = returned_translation - vec<3, double>(translation_we_will_do); //still we can recommend translation in third direction
    return shared_ptr<GeometryObject>(
        new Translation<dim>(const_pointer_cast<ChildType>(dynamic_pointer_cast<const ChildType>(new_child)),
                             this->translation + translation_we_will_do) );*/
}

template <int dim>
void MirrorReflection<dim>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr("axis", axes[direction3D(flipDir)]);
}

template struct MirrorReflection<2>;
template struct MirrorReflection<3>;

/*template <int dim>
shared_ptr<GeometryObject> read_MirrorReflection(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr< MirrorReflection<dim> > res(reader.source.requireAttribute("axis"), new MirrorReflection<dim>());
    return res;
}

static GeometryReader::RegisterObjectReader translation2D_reader(Translation<2>::NAME, read_MirrorReflection<2>);
static GeometryReader::RegisterObjectReader translation3D_reader(Translation<3>::NAME, read_MirrorReflection<3>);*/


}
