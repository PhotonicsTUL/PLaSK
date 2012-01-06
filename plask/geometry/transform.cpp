#include "transform.h"

#include "manager.h"
#include "reader.h"

namespace plask {

template <typename TranslationType>
inline void setupTranslation2d3d(GeometryReader& reader, TranslationType& translation) {
//     translation.translation.x = XML::getAttribiute(source, "x", 0.0);
//     translation.translation.y = XML::getAttribiute(source, "y", 0.0);
    translation.setChild(reader.readExactlyOneChild<typename TranslationType::ChildType>());
}

GeometryElement* read_translation2d(GeometryReader& reader) {
    std::unique_ptr< Translation<2> > translation(new Translation<2>());
    setupTranslation2d3d(reader, *translation);
    return translation.release();
}

GeometryElement* read_translation3d(GeometryReader& reader) {
    std::unique_ptr< Translation<3> > translation(new Translation<3>());
//     translation->translation.z = XML::getAttribiute(source, "z", 0.0);
    setupTranslation2d3d(reader, *translation);
    return translation.release();
}

GeometryReader::RegisterElementReader translation2d_reader("translation2d", read_translation2d);
GeometryReader::RegisterElementReader translation3d_reader("translation3d", read_translation3d);

}   // namespace plask
