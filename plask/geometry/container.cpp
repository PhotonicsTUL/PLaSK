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



StackContainer2d::StackContainer2d(const double baseHeight) {
    stackHeights.push_back(baseHeight);
}

PathHints::Hint StackContainer2d::push_back(StackContainer2d::ChildType* el, const double x_translation) {
    Rect2d bb = el->getBoundingBox();
    const double y_translation = stackHeights.back() - bb.lower.y;
    TranslationT* trans_geom = new TranslationT(el, vec(x_translation, y_translation));
    children.push_back(trans_geom);
    stackHeights.push_back(bb.upper.y + y_translation);
    return PathHints::Hint(this, trans_geom);
}

const plask::StackContainer2d::TranslationT* StackContainer2d::getChildForHeight(double height) const {
    auto it = std::lower_bound(stackHeights.begin(), stackHeights.end(), height);
    if (it == stackHeights.end() || it == stackHeights.begin()) return nullptr;
    return children[it-stackHeights.begin()-1];
}

bool StackContainer2d::inside(const Vec& p) const {
    const TranslationT* c = getChildForHeight(p.y);
    return c ? c->inside(p) : false;
}

shared_ptr<Material> StackContainer2d::getMaterial(const Vec& p) const {
    const TranslationT* c = getChildForHeight(p.y);
    return c ? c->getMaterial(p) : shared_ptr<Material>();
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
                    result.add(&manager.readExactlyOneChild< typename ConstructedType::ChildType >(source));
                }
                
            case irr::io::EXN_COMMENT:
                break;  //skip comments
            
            default:
                throw XMLUnexpectedElementException("<child> or geometry element tag");
            //TODO what with all other XML types (which now are just ignored)?
        }
    }
    throw XMLUnexpectedEndException();
}

GeometryElement* read_TranslationContainer2d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    read_children(*result, manager, source,
        [&]() {
            TranslationContainer<2>::Vec translation;
            translation.x = XML::getAttribiute(source, "x", 0.0);
            translation.y = XML::getAttribiute(source, "y", 0.0);
            return result->add(&manager.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(source), translation);
        }
    );
    return result.release();
}

GeometryElement* read_TranslationContainer3d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< TranslationContainer<3> > result(new TranslationContainer<3>());
    read_children(*result, manager, source,
        [&]() {
            TranslationContainer<3>::Vec translation;
            translation.x = XML::getAttribiute(source, "x", 0.0);
            translation.y = XML::getAttribiute(source, "y", 0.0);
            translation.z = XML::getAttribiute(source, "z", 0.0);
            return result->add(&manager.readExactlyOneChild< typename TranslationContainer<3>::ChildType >(source), translation);
        }
    );
    return result.release();
}

GeometryElement* read_StackContainer2d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< StackContainer2d > result(new StackContainer2d());
    read_children(*result, manager, source,
        [&]() {
            double translation = XML::getAttribiute(source, "x", 0.0);
            return result->add(&manager.readExactlyOneChild< typename StackContainer2d::ChildType >(source), translation);
        }
    );
    return result.release();
}

GeometryManager::RegisterElementReader container2d_reader("container2d", read_TranslationContainer2d);
GeometryManager::RegisterElementReader container3d_reader("container3d", read_TranslationContainer3d);
GeometryManager::RegisterElementReader stack2d_reader("stack2d", read_StackContainer2d);


}	// namespace plask
