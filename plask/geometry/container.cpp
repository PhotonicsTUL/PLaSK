#include "container.h"
#include "../utils/stl.h"

#include "manager.h"

namespace plask {

void PathHints::addHint(const Hint& hint) {
    addHint(hint.first, hint.second);
}

void PathHints::addHint(GeometryElement* container, GeometryElement* child) {
    hintFor[container] = child;
}

GeometryElement* PathHints::getChild(GeometryElement* container) const {
    return map_find(hintFor, container);
}



StackContainer2d::StackContainer2d(const double baseHeight): StackContainerBaseImpl<2>(baseHeight) {}

PathHints::Hint StackContainer2d::push_back(StackContainer2d::ChildType* el, const double tran_translation) {
    Rect2d bb = el->getBoundingBox();
    const double up_translation = stackHeights.back() - bb.lower.up;
    TranslationT* trans_geom = new TranslationT(el, vec(tran_translation, up_translation));
    children.push_back(trans_geom);
    stackHeights.push_back(bb.upper.up + up_translation);
    return PathHints::Hint(this, trans_geom);
}

StackContainer3d::StackContainer3d(const double baseHeight): StackContainerBaseImpl<3>(baseHeight) {}

PathHints::Hint StackContainer3d::push_back(StackContainer3d::ChildType* el, const double lon_translation, const double tran_translation) {
    Rect3d bb = el->getBoundingBox();
    const double up_translation = stackHeights.back() - bb.lower.up;
    TranslationT* trans_geom = new TranslationT(el, vec(lon_translation, tran_translation, up_translation));
    children.push_back(trans_geom);
    stackHeights.push_back(bb.upper.up + up_translation);
    return PathHints::Hint(this, trans_geom);
}

// ---- containers readers: ----

/**
 * Read children, construct ConstructedType::ChildType for each, call child_param_read if children is in \<child\> tag.
 * Read "path" parameter from each \<child\> tag.
 * @param result where add children, must to have add(ConstructedType::ChildType*) method
 * @param manager geometry manager
 * @param source XML data source
 * @param child_param_read call for each \<child\> tag, should create child, add it to container and return PathHints::Hint
 */
template <typename ConstructedType, typename ChildParamF>
void read_children(ConstructedType& result, GeometryManager& manager, XMLReader& source, ChildParamF child_param_read) {

    while (source.read()) {
        switch (source.getNodeType()) {

            case irr::io::EXN_ELEMENT_END:
                return; // container has been read

            case irr::io::EXN_ELEMENT:
                if (source.getNodeName() == std::string("child")) {
                    const char* have_path_name = source.getAttributeValue("path");
                    std::string path = have_path_name;
                    PathHints::Hint hint = child_param_read();
                    if (have_path_name)
                        manager.pathHints[path].addHint(hint);
                } else {
                    result.add(&manager.readElement< typename ConstructedType::ChildType >(source));
                    XML::requireTagEnd(source);
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

GeometryElement* read_TranslationContainer2d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    read_children(*result, manager, source,
        [&]() {
            TranslationContainer<2>::DVec translation;
            translation.c0 = XML::getAttribiute(source, "x", 0.0);
            translation.c1 = XML::getAttribiute(source, "y", 0.0);
            return result->add(&manager.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(source), translation);
        }
    );
    return result.release();
}

GeometryElement* read_TranslationContainer3d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< TranslationContainer<3> > result(new TranslationContainer<3>());
    read_children(*result, manager, source,
        [&]() {
            TranslationContainer<3>::DVec translation;
            translation.c0 = XML::getAttribiute(source, "x", 0.0);
            translation.c1 = XML::getAttribiute(source, "y", 0.0);
            translation.c2 = XML::getAttribiute(source, "z", 0.0);
            return result->add(&manager.readExactlyOneChild< typename TranslationContainer<3>::ChildType >(source), translation);
        }
    );
    return result.release();
}

GeometryElement* read_StackContainer2d(GeometryManager& manager, XMLReader& source) {
    double baseH = XML::getAttribiute(source, "baseHeight", 0.0);
    if (source.getAttributeValue("repeat") == nullptr) {
        std::unique_ptr< StackContainer2d > result(new StackContainer2d(baseH));
        read_children(*result, manager, source,
            [&]() {
                double translation = XML::getAttribiute(source, "x", 0.0);
                return result->add(&manager.readExactlyOneChild< typename StackContainer2d::ChildType >(source), translation);
            }
        );
        return result.release();
    } else {
        unsigned repeat = XML::getAttribiute(source, "repeat", 1);
        std::unique_ptr< MultiStackContainer<2> > result(new MultiStackContainer<2>(baseH, repeat));
        read_children(*result, manager, source,
            [&]() {
                double translation = XML::getAttribiute(source, "x", 0.0);
                return result->add(&manager.readExactlyOneChild< typename StackContainer2d::ChildType >(source), translation);
            }
        );
        return result.release();
    }
}

GeometryManager::RegisterElementReader container2d_reader("container2d", read_TranslationContainer2d);
GeometryManager::RegisterElementReader container3d_reader("container3d", read_TranslationContainer3d);
GeometryManager::RegisterElementReader stack2d_reader("stack2d", read_StackContainer2d);


}	// namespace plask
