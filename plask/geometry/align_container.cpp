#include "align_container.h"

#define align_attr "align"

namespace plask {

template <>
AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::ChildAligner AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::defaultAligner() {
    return align::bottom(0.0);
}

template <>
AlignContainer<2, Primitive<2>::DIRECTION_VERT>::ChildAligner AlignContainer<2, Primitive<2>::DIRECTION_VERT>::defaultAligner() {
    return align::left(0.0);
}

template <>
AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::ChildAligner AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::defaultAligner() {
    return align::bottom(0.0) & align::back(0.0);
}

template <>
AlignContainer<3, Primitive<3>::DIRECTION_VERT>::ChildAligner AlignContainer<3, Primitive<3>::DIRECTION_VERT>::defaultAligner() {
    return align::left(0.0) & align::back(0.0);
}

template <>
AlignContainer<3, Primitive<3>::DIRECTION_LONG>::ChildAligner AlignContainer<3, Primitive<3>::DIRECTION_LONG>::defaultAligner() {
    return align::left(0.0) & align::bottom(0.0);
}


template <int dim, typename Primitive<dim>::Direction alignDirection>
shared_ptr<typename AlignContainer<dim, alignDirection>::TranslationT> AlignContainer<dim, alignDirection>::newTranslation(const shared_ptr<typename AlignContainer<dim, alignDirection>::ChildType>& el, ChildAligner aligner) {
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el);
    align::align(*trans_geom, this->aligner, aligner);
    return trans_geom;
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
shared_ptr<GeometryObject> AlignContainer<dim, alignDirection>::changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< AlignContainer<dim, alignDirection> > result = make_shared< AlignContainer<dim, alignDirection> >(this->getAligner());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first,
                              this->aligners[child_no]);
    return result;
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
PathHints::Hint AlignContainer<dim, alignDirection>::addUnsafe(shared_ptr<AlignContainer<dim, alignDirection>::ChildType> el, ChildAligner aligner) {
    return this->_addUnsafe(newTranslation(el, aligner), aligner);
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
void AlignContainer<dim, alignDirection>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    this->getAligner().writeToXML(dest_xml_object, axes);
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
void AlignContainer<dim, alignDirection>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    childAligners[child_index].writeToXML(dest_xml_child_tag, axes);
}

template struct AlignContainer<2, Primitive<2>::DIRECTION_TRAN>;
template struct AlignContainer<2, Primitive<2>::DIRECTION_VERT>;
template struct AlignContainer<3, Primitive<3>::DIRECTION_LONG>;
template struct AlignContainer<3, Primitive<3>::DIRECTION_TRAN>;
template struct AlignContainer<3, Primitive<3>::DIRECTION_VERT>;

// ---- containers readers: ----

template <Primitive<2>::Direction skipDirection>
inline typename AlignContainer<2, skipDirection>::ChildAligner readPlace(GeometryReader& reader) {
    return align::fromXML(reader.source, *reader.axisNames, align::lowerBoundZero<DirectionWithout<2, skipDirection>::value3d>());
}

template <Primitive<3>::Direction skipDirection>
inline typename AlignContainer<3, skipDirection>::ChildAligner readPlace(GeometryReader& reader) {
    return align::fromXML(reader.source, *reader.axisNames,
                          align::lowerBoundZero<DirectionWithout<3, skipDirection>::valueLower>(),
                          align::lowerBoundZero<DirectionWithout<3, skipDirection>::valueHigher>()
                          );
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
shared_ptr<GeometryObject> read_AlignContainer(GeometryReader& reader, const align::Aligner1D<direction3D(alignDirection)>& aligner) {
    shared_ptr< AlignContainer<dim, alignDirection> > result(new AlignContainer<dim, alignDirection>(aligner));
    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children(reader,
        [&]() -> PathHints::Hint {
            return result->add(reader.readExactlyOneChild< typename AlignContainer<dim, alignDirection>::ChildType >(),
                                readPlace<alignDirection>(reader));
        },
        [&]() {
            result->add(reader.readObject< typename AlignContainer<dim, alignDirection>::ChildType >());
        }
    );
    return result;
}

shared_ptr<GeometryObject> read_AlignContainer2D(GeometryReader& reader) {
    { align::Aligner1D<direction3D(Primitive<2>::Direction(0))> aligner(
          align::fromXML<direction3D(Primitive<2>::Direction(0))>(reader.source, *reader.axisNames));
      if (!aligner.isNull()) return read_AlignContainer<2, Primitive<2>::Direction(0)>(reader, aligner); }
    { align::Aligner1D<direction3D(Primitive<2>::Direction(1))> aligner(
          align::fromXML<direction3D(Primitive<2>::Direction(1))>(reader.source, *reader.axisNames));
      if (!aligner.isNull()) return read_AlignContainer<2, Primitive<2>::Direction(1)>(reader, aligner); }
    throw XMLException(reader.source, "missing aligner description attribute");
}

shared_ptr<GeometryObject> read_AlignContainer3D(GeometryReader& reader) {
    { align::Aligner1D<direction3D(Primitive<3>::Direction(0))> aligner(
          align::fromXML<direction3D(Primitive<3>::Direction(0))>(reader.source, *reader.axisNames));
      if (!aligner.isNull()) return read_AlignContainer<3, Primitive<3>::Direction(0)>(reader, aligner); }
    { align::Aligner1D<direction3D(Primitive<3>::Direction(1))> aligner(
          align::fromXML<direction3D(Primitive<3>::Direction(1))>(reader.source, *reader.axisNames));
      if (!aligner.isNull()) return read_AlignContainer<3, Primitive<3>::Direction(1)>(reader, aligner); }
    { align::Aligner1D<direction3D(Primitive<3>::Direction(2))> aligner(
          align::fromXML<direction3D(Primitive<3>::Direction(2))>(reader.source, *reader.axisNames));
      if (!aligner.isNull()) return read_AlignContainer<3, Primitive<3>::Direction(2)>(reader, aligner); }
    throw XMLException(reader.source, "missing aligner description attribute");
}

static GeometryReader::RegisterObjectReader align_container2D_reader(AlignContainer<2, Primitive<2>::Direction(0)>::NAME, read_AlignContainer2D);
static GeometryReader::RegisterObjectReader align_container3D_reader(AlignContainer<3, Primitive<3>::Direction(0)>::NAME, read_AlignContainer3D);


}   // namespace plask
