#include "container.h"

#include "manager.h"


namespace plask {

// ---- containers readers: ----

shared_ptr<GeometryElement> read_TranslationContainer2d(GeometryReader& reader) {
    shared_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    read_children<TranslationContainer<2>>(reader,
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
    read_children<TranslationContainer<3>>(reader,
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



static GeometryReader::RegisterElementReader container2d_reader("container2d", read_TranslationContainer2d);
static GeometryReader::RegisterElementReader container3d_reader("container3d", read_TranslationContainer3d);


} // namespace plask
