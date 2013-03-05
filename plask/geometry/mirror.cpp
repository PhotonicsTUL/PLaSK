#include "mirror.h"

namespace plask {

template <int dim, typename Primitive<dim>::Direction flipDir>
void MirrorReflection<dim, flipDir>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<Box> result = getChild()->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    for (Box& r: result) dest.push_back(fliped(r));
}

template <int dim, typename Primitive<dim>::Direction flipDir>
void MirrorReflection<dim, flipDir>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    const std::size_t old_size = dest.size();
    getChild()->getPositionsToVec(predicate, dest, path);
    for (std::size_t i = old_size; i < dest.size(); ++i)
        dest[i] = fliped(dest[i]);
}

template <int dim, typename Primitive<dim>::Direction flipDir>
shared_ptr<const GeometryObject> MirrorReflection<dim, flipDir>::changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation) const {
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

template <int dim, typename Primitive<dim>::Direction flipDir>
void MirrorReflection<dim, flipDir>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr("axis", axes[direction3D(flipDir)]);
}

//template struct Translation<2>;
//template struct Translation<3>;


}
