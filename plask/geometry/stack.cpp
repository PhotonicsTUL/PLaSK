#include "stack.h"

namespace plask {

#define baseH_attr "from"
#define repeat_attr "repeat"

shared_ptr<GeometryElement> read_StackContainer2d(GeometryReader& reader) {
    const double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    shared_ptr< StackContainer<2> > result(
                    reader.source.getAttributeValue(repeat_attr) == nullptr ?
                    new StackContainer<2>(baseH) :
                    new MultiStackContainer<2>(baseH, XML::getAttribute(reader.source, repeat_attr, 1))
                );
    read_children(*result, reader,
            [&]() {
                std::unique_ptr<align::Aligner2d<align::DIRECTION_TRAN>> aligner(
                      align::fromStr<align::DIRECTION_TRAN>(XML::getAttribute<std::string>(reader.source, reader.getAxisTranName(), "c")));
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >(), *aligner);
            },
            [&](const shared_ptr<typename StackContainer<2>::ChildType>& child) {
                result->push_front(child);
            }
    );
    return result;
}

shared_ptr<GeometryElement> read_StackContainer3d(GeometryReader& reader) {
    const double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    shared_ptr< StackContainer<3> > result(
                    reader.source.getAttributeValue(repeat_attr) == nullptr ?
                    new StackContainer<3>(baseH) :
                    new MultiStackContainer<3>(baseH, XML::getAttribute(reader.source, repeat_attr, 1))
                );
    read_children(*result, reader,
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

static GeometryReader::RegisterElementReader stack2d_reader("stack2d", read_StackContainer2d);
static GeometryReader::RegisterElementReader stack3d_reader("stack3d", read_StackContainer3d);

}   // namespace plask
