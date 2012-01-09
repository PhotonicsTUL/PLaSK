#include "transform.h"

#include "manager.h"
#include "reader.h"

namespace plask {

template <typename TranslationType>
inline void setupTranslation2d3d(GeometryReader& reader, TranslationType& translation) {
    translation.translation.tran = XML::getAttribiute(reader.source, reader.getAxisTranName(), 0.0);
    translation.translation.up = XML::getAttribiute(reader.source, reader.getAxisUpName(), 0.0);
    translation.setChild(reader.readExactlyOneChild<typename TranslationType::ChildType>());
}

GeometryElement* read_translation2d(GeometryReader& reader) {
    std::unique_ptr< Translation<2> > translation(new Translation<2>());
    setupTranslation2d3d(reader, *translation);
    return translation.release();
}

GeometryElement* read_translation3d(GeometryReader& reader) {
    std::unique_ptr< Translation<3> > translation(new Translation<3>());
    translation->translation.lon = XML::getAttribiute(reader.source, reader.getAxisLonName(), 0.0);
    setupTranslation2d3d(reader, *translation);
    return translation.release();
}

GeometryReader::RegisterElementReader translation2d_reader("translation2d", read_translation2d);
GeometryReader::RegisterElementReader translation3d_reader("translation3d", read_translation3d);

}   // namespace plask
