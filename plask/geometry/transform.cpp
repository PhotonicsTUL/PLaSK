#include "transform.hpp"

#include "../manager.hpp"
#include "reader.hpp"

#define PLASK_TRANSLATION2D_NAME ("translation" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_TRANSLATION3D_NAME ("translation" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)

namespace plask {

template <int dim> const char* Translation<dim>::NAME = dim == 2 ? PLASK_TRANSLATION2D_NAME : PLASK_TRANSLATION3D_NAME;

template <int dim> std::string Translation<dim>::getTypeName() const { return NAME; }

template <int dim>
shared_ptr<Translation<dim>> Translation<dim>::compress(shared_ptr<GeometryObjectD<dim>> child_or_translation,
                                                        const typename Translation<dim>::DVec& translation) {
    shared_ptr<Translation<dim>> as_translation = dynamic_pointer_cast<Translation<dim>>(child_or_translation);
    if (as_translation) {  // translations are compressed, we must create new object because we can't modify
                           // child_or_translation (which can include pointer to objects in original tree)
        return plask::make_shared<Translation<dim>>(as_translation->getChild(),
                                                    as_translation->translation + translation);
    } else {
        return plask::make_shared<Translation<dim>>(child_or_translation, translation);
    }
}

template <int dim> shared_ptr<Material> Translation<dim>::getMaterial(const typename Translation<dim>::DVec& p) const {
    return this->hasChild() ? this->_child->getMaterial(p - translation) : shared_ptr<Material>();
}

template <int dim> bool Translation<dim>::contains(const typename Translation<dim>::DVec& p) const {
    return this->hasChild() ? this->_child->contains(p - translation) : false;
}

template <int dim>
GeometryObject::Subtree Translation<dim>::getPathsAt(const typename Translation::DVec& point, bool all) const {
    if (!this->hasChild()) return GeometryObject::Subtree();
    return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(point - translation, all));
}

template <int dim>
typename Translation<dim>::Box Translation<dim>::fromChildCoords(
    const typename Translation<dim>::ChildType::Box& child_bbox) const {
    return child_bbox.translated(this->translation);
}

template <int dim>
void Translation<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                         std::vector<DVec>& dest,
                                         const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    const std::size_t old_size = dest.size();
    this->_child->getPositionsToVec(predicate, dest, path);
    for (std::size_t i = old_size; i < dest.size(); ++i) dest[i] += translation;
}

template <int dim> shared_ptr<GeometryObject> Translation<dim>::shallowCopy() const { return copyShallow(); }

template <int dim>
shared_ptr<const GeometryObject> Translation<dim>::changedVersion(const GeometryObject::Changer& changer,
                                                                  Vec<3, double>* translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    if (changer.apply(result, translation) || !this->hasChild()) return result;
    Vec<3, double> returned_translation(0.0, 0.0, 0.0);
    shared_ptr<const GeometryObject> new_child = this->getChild()->changedVersion(changer, &returned_translation);
    Vec<dim, double> translation_we_will_do = vec<dim, double>(returned_translation);
    if (new_child == getChild() && translation_we_will_do == Primitive<dim>::ZERO_VEC) return result;
    if (translation)  // we will change translation (partially if dim==2) internaly, so we recommend no extra
                      // translation
        *translation = returned_translation -
                       vec<3, double>(translation_we_will_do);  // still we can recommend translation in third direction
    return shared_ptr<GeometryObject>(
        new Translation<dim>(const_pointer_cast<ChildType>(dynamic_pointer_cast<const ChildType>(new_child)),
                             this->translation + translation_we_will_do));
}

template <> void Translation<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    if (translation.tran() != 0.0) dest_xml_object.attr(axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.0) dest_xml_object.attr(axes.getNameForVert(), translation.vert());
}

template <> void Translation<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    if (translation.lon() != 0.0) dest_xml_object.attr(axes.getNameForLong(), translation.lon());
    if (translation.tran() != 0.0) dest_xml_object.attr(axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.0) dest_xml_object.attr(axes.getNameForVert(), translation.vert());
}

// template <int dim>
// void Translation<dim>::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const
// GeometryObjectD<dim> > >& dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<dim> >(this->shared_from_this()));
//         return;
//     }
//     std::vector< shared_ptr<const GeometryObjectD<dim> > > child_res = getChild()->extract(predicate, path);
//     for (shared_ptr<const GeometryObjectD<dim>>& c: child_res)
//         dest.push_back(Translation<dim>::compress(const_pointer_cast<GeometryObjectD<dim>>(c), this->translation));
// }

template <int dim>
void Translation<dim>::addPointsAlongToSet(std::set<double>& points,
                                      Primitive<3>::Direction direction,
                                      unsigned max_steps,
                                      double min_step_size) const {
    if (this->_child) {
        double trans = translation[int(direction) - (3 - dim)];
        std::set<double> child_points;
        this->_child->addPointsAlongToSet(child_points, direction, this->max_steps ? this->max_steps : max_steps,
                                     this->min_step_size ? this->min_step_size : min_step_size);
        for (double p : child_points) points.insert(p + trans);
    }
}

template <int dim>
void Translation<dim>::addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                                            unsigned max_steps,
                                            double min_step_size) const {
    if (this->_child) {
        std::set<typename GeometryObjectD<dim>::LineSegment> child_segments;
        this->_child->addLineSegmentsToSet(child_segments, this->max_steps ? this->max_steps : max_steps,
                                           this->min_step_size ? this->min_step_size : min_step_size);
        for (const auto& p : child_segments)
            segments.insert(typename GeometryObjectD<dim>::LineSegment(p[0] + translation, p[1] + translation));
    }
}

template <typename TranslationType>
inline static void setupTranslation2D3D(GeometryReader& reader, TranslationType& translation) {
    translation.translation.tran() = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
    translation.translation.vert() = reader.source.getAttribute(reader.getAxisVertName(), 0.0);
    translation.setChild(reader.readExactlyOneChild<typename TranslationType::ChildType>(!reader.manager.draft));
}

shared_ptr<GeometryObject> read_translation2D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    shared_ptr<Translation<2>> translation(new Translation<2>());
    setupTranslation2D3D(reader, *translation);
    return translation;
}

shared_ptr<GeometryObject> read_translation3D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr<Translation<3>> translation(new Translation<3>());
    translation->translation.lon() = reader.source.getAttribute(reader.getAxisLongName(), 0.0);
    setupTranslation2D3D(reader, *translation);
    return translation;
}

static GeometryReader::RegisterObjectReader translation2D_reader(PLASK_TRANSLATION2D_NAME, read_translation2D);
static GeometryReader::RegisterObjectReader translation3D_reader(PLASK_TRANSLATION3D_NAME, read_translation3D);

template struct PLASK_API Translation<2>;
template struct PLASK_API Translation<3>;

}  // namespace plask
