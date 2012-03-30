#include "stack.h"

namespace plask {

#define baseH_attr "from"
#define repeat_attr "repeat"
#define extend_attr "extend"

template <int dim>
inline static typename StackContainer<dim>::StackExtension readStackExtension(GeometryReader& reader) {
    boost::optional<std::string> extend_opt = XML::getAttribute(reader.source, extend_attr);
    if (extend_opt) {
        std::string extend_str = *extend_opt;
        for (auto s: extend_str) s = std::tolower(s);
        if (extend_str == "vert" || extend_str == "vertical" || extend_str == "vertically")
            return StackContainer<dim>::EXTEND_VERTICALLY;
        else if (extend_str == "hor" || extend_str == "horizontal" || extend_str == "horizontally")
            return StackContainer<dim>::EXTEND_HORIZONTALLY;
        else if (extend_str == "all" || extend_str == "both")
            return StackContainer<dim>::EXTEND_ALL;
        else throw XMLUnexpectedAttributeValueException("stack", extend_attr, extend_str);
    }
    return StackContainer<dim>::EXTEND_NONE;
}

shared_ptr<GeometryElement> read_StackContainer2d(GeometryReader& reader) {
    const double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    std::unique_ptr<align::Aligner2d<align::DIRECTION_TRAN>> default_aligner(
          align::fromStr<align::DIRECTION_TRAN>(XML::getAttribute<std::string>(reader.source, reader.getAxisTranName(), "c")));

    auto extend = readStackExtension<2>(reader);
    auto repeat = reader.source.getAttributeValue(repeat_attr);
    if (repeat != nullptr && extend != StackContainer<2>::EXTEND_NONE)
        throw XMLConflictingAttributesException("stack", "repeat", "extend");

    shared_ptr< StackContainer<2> > result(
                    repeat == nullptr ?
                    new StackContainer<2>(baseH, extend) :
                    new MultiStackContainer<2>(XML::getAttribute(reader.source, repeat_attr, 1), baseH)
                );
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<StackContainer<2>>(reader,
            [&]() {
                boost::optional<std::string> aligner_str = XML::getAttribute(reader.source, reader.getAxisTranName());
                if (aligner_str) {
                   std::unique_ptr<align::Aligner2d<align::DIRECTION_TRAN>> aligner(align::fromStr<align::DIRECTION_TRAN>(*aligner_str));
                   return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >(), *aligner);
                } else {
                   return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >(), *default_aligner);
                }
            },
            [&](const shared_ptr<typename StackContainer<2>::ChildType>& child) {
                result->push_front(child);
            }
    );
    return result;
}

shared_ptr<GeometryElement> read_StackContainer3d(GeometryReader& reader) {
    const double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    //TODO default aligner (see above)

    auto extend = readStackExtension<3>(reader);
    auto repeat = reader.source.getAttributeValue(repeat_attr);
    if (repeat != nullptr && extend != StackContainer<3>::EXTEND_NONE)
        throw XMLConflictingAttributesException("stack", "repeat", "extend");

    shared_ptr< StackContainer<3> > result(
                    repeat == nullptr ?
                    new StackContainer<3>(baseH, extend) :
                    new MultiStackContainer<3>(XML::getAttribute(reader.source, repeat_attr, 1), baseH)
                );

    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children<StackContainer<3>>(reader,
            [&]() {
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<3>::ChildType >(),
                                          align::fromStr<align::DIRECTION_LON, align::DIRECTION_TRAN>(
                                              XML::getAttribute<std::string>(reader.source, reader.getAxisLonName(), "c"),
                                              XML::getAttribute<std::string>(reader.source, reader.getAxisTranName(), "c")
                                          ));
            },
            [&](const shared_ptr<typename StackContainer<3>::ChildType>& child) {
                result->push_front(child);
            }
    );
    return result;
}

static GeometryReader::RegisterElementReader stack2d_reader("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_StackContainer2d);
static GeometryReader::RegisterElementReader stack3d_reader("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_StackContainer3d);

}   // namespace plask
