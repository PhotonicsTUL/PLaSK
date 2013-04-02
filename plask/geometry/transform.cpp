#include "transform.h"

#include "reader.h"

namespace plask {

template <int dim>
shared_ptr<Translation<dim>> Translation<dim>::compress(shared_ptr<GeometryObjectD<dim> > child_or_translation, const Translation<dim>::DVec &translation) {
    shared_ptr< Translation<dim> > as_translation = dynamic_pointer_cast< Translation<dim> >(child_or_translation);
    if (as_translation) {    // translations are compressed, we must create new object because we can't modify child_or_translation (which can include pointer to objects in original tree)
        return make_shared< Translation<dim> >(as_translation->getChild(), as_translation->translation + translation);
    } else {
        return make_shared< Translation<dim> >(child_or_translation, translation);
    }
}

template <int dim>
void Translation<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<Box> result = getChild()->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    for (Box& r: result) dest.push_back(r.translated(translation));
}

template <int dim>
void Translation<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
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
shared_ptr<const GeometryObject> Translation<dim>::changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    if (changer.apply(result, translation) || !this->hasChild()) return result;
    Vec<3, double> returned_translation(0.0, 0.0, 0.0);
    shared_ptr<const GeometryObject> new_child = this->getChild()->changedVersion(changer, &returned_translation);
    Vec<dim, double> translation_we_will_do = vec<dim, double>(returned_translation);
    if (new_child == getChild() && translation_we_will_do == Primitive<dim>::ZERO_VEC) return result;
    if (translation)    //we will change translation (partially if dim==2) internaly, so we recommend no extra translation
        *translation = returned_translation - vec<3, double>(translation_we_will_do); //still we can recommend translation in third direction
    return shared_ptr<GeometryObject>(
        new Translation<dim>(const_pointer_cast<ChildType>(dynamic_pointer_cast<const ChildType>(new_child)),
                             this->translation + translation_we_will_do) );
}

template <>
void Translation<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    if (translation.tran() != 0.0) dest_xml_object.attr(axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.0) dest_xml_object.attr(axes.getNameForVert(), translation.vert());
}

template <>
void Translation<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    if (translation.lon() != 0.0) dest_xml_object.attr(axes.getNameForLong(), translation.lon());
    if (translation.tran() != 0.0) dest_xml_object.attr(axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.0) dest_xml_object.attr(axes.getNameForVert(), translation.vert());
}

// template <int dim>
// void Translation<dim>::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<dim> >(this->shared_from_this()));
//         return;
//     }
//     std::vector< shared_ptr<const GeometryObjectD<dim> > > child_res = getChild()->extract(predicate, path);
//     for (shared_ptr<const GeometryObjectD<dim>>& c: child_res)
//         dest.push_back(Translation<dim>::compress(const_pointer_cast<GeometryObjectD<dim>>(c), this->translation));
// }

template struct Translation<2>;
template struct Translation<3>;

template <typename TranslationType>
inline static void setupTranslation2D3D(GeometryReader& reader, TranslationType& translation) {
    translation.translation.tran() = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
    translation.translation.vert() = reader.source.getAttribute(reader.getAxisVertName(), 0.0);
    translation.setChild(reader.readExactlyOneChild<typename TranslationType::ChildType>());
}

shared_ptr<GeometryObject> read_translation2D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    shared_ptr< Translation<2> > translation(new Translation<2>());
    setupTranslation2D3D(reader, *translation);
    return translation;
}

shared_ptr<GeometryObject> read_translation3D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr< Translation<3> > translation(new Translation<3>());
    translation->translation.lon() = reader.source.getAttribute(reader.getAxisLongName(), 0.0);
    setupTranslation2D3D(reader, *translation);
    return translation;
}

static GeometryReader::RegisterObjectReader translation2D_reader(Translation<2>::NAME, read_translation2D);
static GeometryReader::RegisterObjectReader translation3D_reader(Translation<3>::NAME, read_translation3D);

}   // namespace plask
