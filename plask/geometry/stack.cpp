#include "stack.h"

namespace plask {

#define baseH_attr "from"
#define repeat_attr "repeat"

shared_ptr<GeometryElement> read_StackContainer2d(GeometryReader& reader) {
    double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    if (reader.source.getAttributeValue(repeat_attr) == nullptr) {
        shared_ptr< StackContainer<2> > result(new StackContainer<2>(baseH));
        read_children(*result, reader,
            [&]() {
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >());
//TODO                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            },
            [&](const shared_ptr<typename StackContainer<2>::ChildType>& child) {
                result->push_front(child);
            }
        );
        return result;
    } else {
        unsigned repeat = XML::getAttribute(reader.source, repeat_attr, 1);
        shared_ptr< MultiStackContainer<2> > result(new MultiStackContainer<2>(baseH, repeat));
        read_children(*result, reader,
            [&]() {
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >());
//TODO                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            },
            [&](const shared_ptr<typename MultiStackContainer<2>::ChildType>& child) {
                result->push_front(child);
            }
        );
        return result;
    }
}

shared_ptr<GeometryElement> read_StackContainer3d(GeometryReader& reader) {
    double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    if (reader.source.getAttributeValue(repeat_attr) == nullptr) {
        shared_ptr< StackContainer<3> > result(new StackContainer<3>(baseH));
        read_children(*result, reader,
           [&]() {
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<3>::ChildType >());
//TODO                                   XML::getAttribute(reader.source, reader.getAxisLonName(), 0.0),
//TODO                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            },
            [&](const shared_ptr<typename StackContainer<3>::ChildType>& child) {
                result->push_front(child);
            }
        );
        return result;
    } else {
        unsigned repeat = XML::getAttribute(reader.source, repeat_attr, 1);
        shared_ptr< MultiStackContainer<3> > result(new MultiStackContainer<3>(baseH, repeat));
        read_children(*result, reader,
            [&]() {
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<3>::ChildType >());
//TODO                                   XML::getAttribute(reader.source, reader.getAxisLonName(), 0.0),
//TODO                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            },
            [&](const shared_ptr<typename MultiStackContainer<3>::ChildType>& child) {
                result->push_front(child);
            }
        );
        return result;
    }
}

static GeometryReader::RegisterElementReader stack2d_reader("stack2d", read_StackContainer2d);
static GeometryReader::RegisterElementReader stack3d_reader("stack3d", read_StackContainer3d);

}   // namespace plask
