#include "transform.h"

#include "manager.h"
#include "reader.h"

namespace plask {

template <int dim>
void Translation<dim>::getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<Box> result = getChild()->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    for (Box& r: result) dest.push_back(r.translated(translation));
}

template <int dim>
void Translation<dim>::getPositionsToVec(const GeometryElement::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    const std::size_t old_size = dest.size();
    getChild()->getPositionsToVec(predicate, dest, path);
    for (std::size_t i = old_size; i < dest.size(); ++i)
        dest[i] += translation;
}

template <int dim>
shared_ptr<const GeometryElement> Translation<dim>::changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation) const {
    shared_ptr<const GeometryElement> result(this->shared_from_this());
    if (changer.apply(result, translation) || !this->hasChild()) return result;
    Vec<3, double> returned_translation(0.0, 0.0, 0.0);
    shared_ptr<const GeometryElement> new_child = this->getChild()->changedVersion(changer, &returned_translation);
    Vec<dim, double> translation_we_will_do = vec<dim, double>(returned_translation);
    if (new_child == getChild() && translation_we_will_do == Primitive<dim>::ZERO_VEC) return result;
    if (translation)    //we will change translation (partially if dim==2) internaly, so we recommend no extra translation
        *translation = returned_translation - vec<3, double>(translation_we_will_do); //still we can recommend translation in third direction
    return shared_ptr<GeometryElement>(
        new Translation<dim>(const_pointer_cast<ChildType>(dynamic_pointer_cast<const ChildType>(new_child)),
        this->translation + translation_we_will_do) );
}

template class Translation<2>;
template class Translation<3>;     

template <typename TranslationType>
inline void setupTranslation2d3d(GeometryReader& reader, TranslationType& translation) {
    translation.translation.tran = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
    translation.translation.up = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
    translation.setChild(reader.readExactlyOneChild<typename TranslationType::ChildType>());
}

shared_ptr<GeometryElement> read_translation2d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    shared_ptr< Translation<2> > translation(new Translation<2>());
    setupTranslation2d3d(reader, *translation);
    return translation;
}

shared_ptr<GeometryElement> read_translation3d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr< Translation<3> > translation(new Translation<3>());
    translation->translation.lon = reader.source.getAttribute(reader.getAxisLonName(), 0.0);
    setupTranslation2d3d(reader, *translation);
    return translation;
}

static GeometryReader::RegisterElementReader translation2d_reader("translation" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_translation2d);
static GeometryReader::RegisterElementReader translation3d_reader("translation" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_translation3d);

}   // namespace plask
