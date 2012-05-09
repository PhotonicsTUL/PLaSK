#include "container.h"

#include "manager.h"


namespace plask {

template <int dim>
bool GeometryElementContainer<dim>::childrenEraseFromEnd(typename TranslationVector::iterator firstToErase) {
    if (firstToErase != children.end()) {
        children.erase(firstToErase, children.end());
        fireChildrenChanged();
        return true;
    } else
        return false;
}

template <int dim>
bool GeometryElementContainer<dim>::removeT(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    auto dst = children.begin();
    for (auto i: children)
        if (predicate(i))
            disconnectOnChildChanged(*i);
        else
            *dst++ = i;
    return childrenEraseFromEnd(dst);
}

template class GeometryElementContainer<2>;
template class GeometryElementContainer<3>;

// ---- containers readers: ----

shared_ptr<GeometryElement> read_TranslationContainer2d(GeometryReader& reader) {
    shared_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<TranslationContainer<2>>(reader,
        [&]() {
            TranslationContainer<2>::DVec translation;
            translation.tran = reader.source.getAttribute(reader.getAxisLonName(), 0.0);
            translation.up = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
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
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children<TranslationContainer<3>>(reader,
        [&]() {
            TranslationContainer<3>::DVec translation;
            translation.c0 = reader.source.getAttribute(reader.getAxisName(0), 0.0);
            translation.c1 = reader.source.getAttribute(reader.getAxisName(1), 0.0);
            translation.c2 = reader.source.getAttribute(reader.getAxisName(2), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<3>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<3>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}



static GeometryReader::RegisterElementReader container2d_reader("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_TranslationContainer2d);
static GeometryReader::RegisterElementReader container3d_reader("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_TranslationContainer3d);


} // namespace plask
