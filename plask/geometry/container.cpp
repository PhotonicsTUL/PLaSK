#include "container.h"

#include "manager.h"
#include "reader.h"

namespace plask {

// ---- containers readers: ----

/**
 * Read children, construct ConstructedType::ChildType for each, call child_param_read if children is in \<child\> tag.
 * Read "path" parameter from each \<child\> tag.
 * @param reader reader
 * @param source XML data source
 * @param child_param_read call for each \<child\> tag, should create child, add it to container and return PathHints::Hint
 */
template <typename ConstructedType, typename ChildParamF, typename WithoutChildParamF>
void read_children(ConstructedType& result, GeometryReader& reader, ChildParamF child_param_read, WithoutChildParamF without_child_param_add) {

    std::string container_tag_name = reader.source.getNodeName();

    while (reader.source.read()) {
        switch (reader.source.getNodeType()) {

            case irr::io::EXN_ELEMENT_END:
                if (reader.source.getNodeName() != container_tag_name)
                    throw XMLUnexpectedElementException("end of \"" + container_tag_name + "\" tag");
                return; // container has been read

            case irr::io::EXN_ELEMENT:
                if (reader.source.getNodeName() == std::string("child")) {
                    boost::optional<std::string> path = XML::getAttribute(reader.source, "path");
                    PathHints::Hint hint = child_param_read();
                    if (path)
                        reader.manager.pathHints[*path].addHint(hint);  //this call readExactlyOneChild
                } else {
                    without_child_param_add(reader.readElement< typename ConstructedType::ChildType >());

                    //std::string element_tag_name = reader.source.getNodeName();
                    //result.add(reader.readElement< typename ConstructedType::ChildType >());
                    //XML::requireTagEndOrEmptyTag(reader.source, element_tag_name);
                    //result.add(&manager.readExactlyOneChild< typename ConstructedType::ChildType >(source));
                }

            case irr::io::EXN_COMMENT:
                break;  //skip comments

            default:
                throw XMLUnexpectedElementException("<child> or geometry element tag");
        }
    }
    throw XMLUnexpectedEndException();
}

shared_ptr<GeometryElement> read_TranslationContainer2d(GeometryReader& reader) {
    shared_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    read_children(*result, reader,
        [&]() {
            TranslationContainer<2>::DVec translation;
            translation.tran = XML::getAttribute(reader.source, reader.getAxisLonName(), 0.0);
            translation.up = XML::getAttribute(reader.source, reader.getAxisUpName(), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<2>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}

shared_ptr<GeometryElement> read_TranslationContainer3d(GeometryReader& reader) {
    shared_ptr< TranslationContainer<3> > result(new TranslationContainer<3>());
    read_children(*result, reader,
        [&]() {
            TranslationContainer<3>::DVec translation;
            translation.c0 = XML::getAttribute(reader.source, reader.getAxisName(0), 0.0);
            translation.c1 = XML::getAttribute(reader.source, reader.getAxisName(1), 0.0);
            translation.c2 = XML::getAttribute(reader.source, reader.getAxisName(2), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<3>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<3>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}

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

static GeometryReader::RegisterElementReader container2d_reader("container2d", read_TranslationContainer2d);
static GeometryReader::RegisterElementReader container3d_reader("container3d", read_TranslationContainer3d);
static GeometryReader::RegisterElementReader stack2d_reader("stack2d", read_StackContainer2d);
static GeometryReader::RegisterElementReader stack3d_reader("stack3d", read_StackContainer3d);


} // namespace plask
