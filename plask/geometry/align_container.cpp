#include "align_container.h"

namespace plask {

template<>
shared_ptr<AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::TranslationT> AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::newTranslation(
        const shared_ptr<AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::ChildType>& el, const double& place) {
    return make_shared<TranslationT>(el, vec(0.0, place));
}

template<>
shared_ptr<AlignContainer<2, Primitive<2>::DIRECTION_VERT>::TranslationT> AlignContainer<2, Primitive<2>::DIRECTION_VERT>::newTranslation(
        const shared_ptr<AlignContainer<2, Primitive<2>::DIRECTION_VERT>::ChildType>& el, const double& place) {
    return make_shared<TranslationT>(el, vec(place, 0.0));
}

template<>
shared_ptr<AlignContainer<3, Primitive<3>::DIRECTION_LONG>::TranslationT> AlignContainer<3, Primitive<3>::DIRECTION_LONG>::newTranslation(
        const shared_ptr<AlignContainer<3, Primitive<3>::DIRECTION_LONG>::ChildType>& el, const std::pair<double, double>& place) {
    return make_shared<TranslationT>(el, vec(0.0, place.first, place.second));
}

template<>
shared_ptr<AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::TranslationT> AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::newTranslation(
        const shared_ptr<AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::ChildType>& el, const std::pair<double, double>& place) {
    return make_shared<TranslationT>(el, vec(place.first, 0.0, place.second));
}

template<>
shared_ptr<AlignContainer<3, Primitive<3>::DIRECTION_VERT>::TranslationT> AlignContainer<3, Primitive<3>::DIRECTION_VERT>::newTranslation(
        const shared_ptr<AlignContainer<3, Primitive<3>::DIRECTION_VERT>::ChildType>& el, const std::pair<double, double>& place) {
    return make_shared<TranslationT>(el, vec(place.first, place.second, 0.0));
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
shared_ptr<typename AlignContainer<dim, alignDirection>::TranslationT> AlignContainer<dim, alignDirection>::newChild(const shared_ptr<typename AlignContainer<dim, alignDirection>::ChildType>& el, const AlignContainer<dim, alignDirection>::Coordinates& place) {
    shared_ptr<AlignContainer<dim, alignDirection>::TranslationT> trans_geom = this->newTranslation(el, place);
    this->aligner->align(*trans_geom);
    this->connectOnChildChanged(*trans_geom);
    return trans_geom;
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
PathHints::Hint AlignContainer<dim, alignDirection>::addUnsafe(const shared_ptr<AlignContainer<dim, alignDirection>::ChildType>& el, const AlignContainer<dim, alignDirection>::Coordinates& place) {
    shared_ptr<AlignContainer<dim, alignDirection>::TranslationT> trans_geom = this->newChild(el, place);
    this->children.push_back(trans_geom);
    this->fireChildrenInserted(children.size()-1, children.size());
    return PathHints::Hint(shared_from_this(), trans_geom);
}

template struct AlignContainer<2, Primitive<2>::DIRECTION_TRAN>;
template struct AlignContainer<2, Primitive<2>::DIRECTION_VERT>;
template struct AlignContainer<3, Primitive<3>::DIRECTION_LONG>;
template struct AlignContainer<3, Primitive<3>::DIRECTION_TRAN>;
template struct AlignContainer<3, Primitive<3>::DIRECTION_VERT>;

// ---- containers readers: ----

/*template <Primitive<2>::Direction alignDirection>
shared_ptr<GeometryObject> read_AlignContainer2D(GeometryReader& reader) {

    shared_ptr< AlignContainer<2, alignDirection> > result(new AlignContainer<2, alignDirection>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children(reader,
        [&]() -> PathHints::Hint {
            TranslationContainer<2>::DVec translation;
            translation.tran() = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
            translation.vert() = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(), translation);
        },
        [&]() {
            result->add(reader.readObject< typename TranslationContainer<2>::ChildType >());
        }
    );
    return result;
}

shared_ptr<GeometryObject> read_TranslationContainer3D(GeometryReader& reader) {
    shared_ptr< TranslationContainer<3> > result(new TranslationContainer<3>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children(reader,
        [&]() -> PathHints::Hint {
            TranslationContainer<3>::DVec translation;
            translation.c0 = reader.source.getAttribute(reader.getAxisName(0), 0.0);
            translation.c1 = reader.source.getAttribute(reader.getAxisName(1), 0.0);
            translation.c2 = reader.source.getAttribute(reader.getAxisName(2), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<3>::ChildType >(), translation);
        },
        [&]() {
            result->add(reader.readObject< typename TranslationContainer<3>::ChildType >());
        }
    );
    return result;
}

static GeometryReader::RegisterObjectReader container2D_reader(TranslationContainer<2>::NAME, read_TranslationContainer2D);
static GeometryReader::RegisterObjectReader container3D_reader(TranslationContainer<3>::NAME, read_TranslationContainer3D);*/

}   // namespace plask
