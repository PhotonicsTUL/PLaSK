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

PathHints::Hint StackContainer2d::push_back(StackContainer2d::ChildT* el, const double x_translation) {
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

GeometryElement* read_TranslationContainer2d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    while (source.read()) {
        switch (source.getNodeType()) {
            // container is read
            case irr::io::EXN_ELEMENT_END:
                return result.release();  

            case irr::io::EXN_ELEMENT:
                if (source.getNodeName() == std::string("child")) {
                    TranslationContainer<2>::Vec translation;
                    translation.x = XML::getAttribiute(source, "x", 0.0);
                    translation.y = XML::getAttribiute(source, "y", 0.0);
                    const char* have_path_name = source.getAttributeValue("path");
                    std::string path = have_path_name;
                    PathHints::Hint hint = result->add(&manager.readExactlyOneChild< GeometryElementD<2> >(source), translation);
                    if (have_path_name)
                        manager.pathHints[path].addHint(hint);
                } else {
                    result->add(&manager.readExactlyOneChild< GeometryElementD<2> >(source));
                }
            
            default:    ;
            //TODO what with all other XML types (which now are just ignored)?
        }
    }
    throw XMLUnexpectedEndException();
}



}	// namespace plask
