#include "transform.h"

#include "manager.h"

namespace plask {

template <typename TranslationType>
inline void setupTranslation2d3d(GeometryManager& manager, XMLReader& source, TranslationType& translation) {
    translation.translation.x = XML::getAttribiute(source, "x", 0.0);
    translation.translation.y = XML::getAttribiute(source, "y", 0.0);
    translation.setChild(manager.readExactlyOneChild<typename TranslationType::ChildType>(source));
}

GeometryElement* read_translation2d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< Translation<2> > translation(new Translation<2>());
    setupTranslation2d3d(manager, source, *translation);
    return translation.release();
}

GeometryElement* read_translation3d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< Translation<3> > translation(new Translation<3>());
    translation->translation.z = XML::getAttribiute(source, "z", 0.0);
    setupTranslation2d3d(manager, source, *translation);
    return translation.release();
}

GeometryManager::RegisterElementReader translation2d_reader("translation2d", read_translation2d);
GeometryManager::RegisterElementReader translation3d_reader("translation3d", read_translation3d);

}   // namespace plask
