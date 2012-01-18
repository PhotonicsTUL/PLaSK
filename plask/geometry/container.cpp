#include "container.h"

#include "manager.h"
#include "reader.h"

namespace plask {

void PathHints::addHint(const Hint& hint) {
    addHint(hint.first, hint.second);
}

void PathHints::addHint(weak_ptr<GeometryElement> container, weak_ptr<GeometryElement> child) {
    hintFor[container] = child;
}

shared_ptr<GeometryElement> PathHints::getChild(shared_ptr<GeometryElement> container) {
    auto e = hintFor.find(container);
    if (e == hintFor.end()) return shared_ptr<GeometryElement>();
    shared_ptr<GeometryElement> result = e->second.lock();
    if (!result || e->first.expired()) {        //child or container was deleted (in second case, new container is under same address as was old one)
        hintFor.erase(container);
        return shared_ptr<GeometryElement>();
    }
    return result;
}

shared_ptr<GeometryElement> PathHints::getChild(shared_ptr<GeometryElement> container) const {
    auto e = hintFor.find(container);
    return (e == hintFor.end() || e->first.expired()) ? shared_ptr<GeometryElement>() : e->second.lock();
}

void PathHints::cleanDeleted() {
    for(auto i = hintFor.begin(); i != hintFor.end(); ) {
        if (i->first.expired() || i->second.expired())
            hintFor.erase(i++);
        else
            ++i;
}

}

StackContainer2d::StackContainer2d(const double baseHeight): StackContainerBaseImpl<2>(baseHeight) {}

PathHints::Hint StackContainer2d::add(const shared_ptr<StackContainerBaseImpl::ChildType> &el, const double tran_translation) {
    ensureCanHasAsChild(*el);
    return addUnsafe(el, tran_translation);
}

PathHints::Hint StackContainer2d::addUnsafe(const shared_ptr<ChildType>& el, const double tran_translation) {
    double el_translation, next_height;
    calcHeight(el, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom(new TranslationT(el, vec(tran_translation, el_translation)));
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);
    return PathHints::Hint(shared_from_this(), trans_geom);
}

StackContainer3d::StackContainer3d(const double baseHeight): StackContainerBaseImpl<3>(baseHeight) {}

PathHints::Hint StackContainer3d::add(const shared_ptr<ChildType>& el, const double lon_translation, const double tran_translation) {
    ensureCanHasAsChild(*el);
    return addUnsafe(el, lon_translation, tran_translation);
}

PathHints::Hint StackContainer3d::addUnsafe(const shared_ptr<ChildType>& el, const double lon_translation, const double tran_translation) {
    double el_translation, next_height;
    calcHeight(el, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom(new TranslationT(el, vec(lon_translation, tran_translation, el_translation)));
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);
    return PathHints::Hint(shared_from_this(), trans_geom);
}

// ---- containers readers: ----

/**
 * Read children, construct ConstructedType::ChildType for each, call child_param_read if children is in \<child\> tag.
 * Read "path" parameter from each \<child\> tag.
 * @param reader reader
 * @param source XML data source
 * @param child_param_read call for each \<child\> tag, should create child, add it to container and return PathHints::Hint
 */
template <typename ConstructedType, typename ChildParamF>
void read_children(ConstructedType& result, GeometryReader& reader, ChildParamF child_param_read) {

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
                    //std::string element_tag_name = reader.source.getNodeName();
                    result.add(reader.readElement< typename ConstructedType::ChildType >());
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
        }
    );
    return result;
}

#define baseH_attr "from"
#define repeat_attr "repeat"

shared_ptr<GeometryElement> read_StackContainer2d(GeometryReader& reader) {
    double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    if (reader.source.getAttributeValue(repeat_attr) == nullptr) {
        shared_ptr< StackContainer2d > result(new StackContainer2d(baseH));
        read_children(*result, reader,
            [&]() {
                return result->add(reader.readExactlyOneChild< typename StackContainer2d::ChildType >(),
                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            }
        );
        return result;
    } else {
        unsigned repeat = XML::getAttribute(reader.source, repeat_attr, 1);
        shared_ptr< MultiStackContainer<2> > result(new MultiStackContainer<2>(baseH, repeat));
        read_children(*result, reader,
            [&]() {
                return result->add(reader.readExactlyOneChild< typename StackContainer2d::ChildType >(),
                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            }
        );
        return result;
    }
}

shared_ptr<GeometryElement> read_StackContainer3d(GeometryReader& reader) {
    double baseH = XML::getAttribute(reader.source, baseH_attr, 0.0);
    if (reader.source.getAttributeValue(repeat_attr) == nullptr) {
        shared_ptr< StackContainer3d > result(new StackContainer3d(baseH));
        read_children(*result, reader,
           [&]() {
                return result->add(reader.readExactlyOneChild< typename StackContainer3d::ChildType >(),
                                   XML::getAttribute(reader.source, reader.getAxisLonName(), 0.0),
                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            }
        );
        return result;
    } else {
        unsigned repeat = XML::getAttribute(reader.source, repeat_attr, 1);
        shared_ptr< MultiStackContainer<3> > result(new MultiStackContainer<3>(baseH, repeat));
        read_children(*result, reader,
            [&]() {
                return result->add(reader.readExactlyOneChild< typename StackContainer3d::ChildType >(),
                                   XML::getAttribute(reader.source, reader.getAxisLonName(), 0.0),
                                   XML::getAttribute(reader.source, reader.getAxisTranName(), 0.0));
            }
        );
        return result;
    }
}

GeometryReader::RegisterElementReader container2d_reader("container2d", read_TranslationContainer2d);
GeometryReader::RegisterElementReader container3d_reader("container3d", read_TranslationContainer3d);
GeometryReader::RegisterElementReader stack2d_reader("stack2d", read_StackContainer2d);
GeometryReader::RegisterElementReader stack3d_reader("stack3d", read_StackContainer3d);


} // namespace plask
