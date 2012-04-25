#include "transform.h"

#include "manager.h"
#include "reader.h"

namespace plask {

template <typename TranslationType>
inline void setupTranslation2d3d(GeometryReader& reader, TranslationType& translation) {
    translation.translation.tran = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
    translation.translation.up = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
    translation.setChild(reader.readExactlyOneChild<typename TranslationType::ChildType>());
}

shared_ptr<GeometryElement> read_translation2d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    shared_ptr< Translation<2> > translation(new Translation<2>());
    setupTranslation2d3d(reader, *translation);
    return translation;
}

shared_ptr<GeometryElement> read_translation3d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr< Translation<3> > translation(new Translation<3>());
    translation->translation.lon = reader.source.getAttribute(reader.getAxisLonName(), 0.0);
    setupTranslation2d3d(reader, *translation);
    return translation;
}

static GeometryReader::RegisterElementReader translation2d_reader("translation" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_translation2d);
static GeometryReader::RegisterElementReader translation3d_reader("translation" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_translation3d);

}   // namespace plask
