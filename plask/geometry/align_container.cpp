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
shared_ptr<typename AlignContainer<dim, alignDirection>::TranslationT> AlignContainer<dim, alignDirection>::newChild(const shared_ptr<typename AlignContainer<dim, alignDirection>::ChildType>& el, const Vec<dim, double>& translation) {
    shared_ptr<AlignContainer<dim, alignDirection>::TranslationT> trans_geom = make_shared<TranslationT>(el, translation);
    this->aligner->align(*trans_geom);
    this->connectOnChildChanged(*trans_geom);
    return trans_geom;
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
shared_ptr<GeometryObject> AlignContainer<dim, alignDirection>::changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< AlignContainer<dim, alignDirection> > result = make_shared< AlignContainer<dim, alignDirection> >(this->getAligner());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first, children[child_no]->translation + vec<dim, double>(children_after_change[child_no].second));
    return result;
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
PathHints::Hint AlignContainer<dim, alignDirection>::addUnsafe(shared_ptr<AlignContainer<dim, alignDirection>::ChildType> el, const AlignContainer<dim, alignDirection>::Coordinates& place) {
    shared_ptr<AlignContainer<dim, alignDirection>::TranslationT> trans_geom = this->newChild(el, place);
    this->children.push_back(trans_geom);
    this->fireChildrenInserted(children.size()-1, children.size());
    return PathHints::Hint(shared_from_this(), trans_geom);
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
PathHints::Hint AlignContainer<dim, alignDirection>::addUnsafe(shared_ptr<AlignContainer<dim, alignDirection>::ChildType> el, const Vec<dim, double>& translation) {
    shared_ptr<AlignContainer<dim, alignDirection>::TranslationT> trans_geom = this->newChild(el, translation);
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

inline double readPlace(GeometryReader& reader, Primitive<2>::Direction skipDirection) {
    return reader.source.getAttribute(reader.getAxisName(1-skipDirection), 0.0);
}

inline Vec<3, double> readPlace(GeometryReader& reader, Primitive<3>::Direction skipDirection) {
    Vec<3, double> result;
    for (int i = 0; i < 3; ++i)
        if (i != skipDirection)
            result[i] = reader.source.getAttribute(reader.getAxisName(i), 0.0);
    return result;
}

template <int dim, typename Primitive<dim>::Direction alignDirection>
shared_ptr<GeometryObject> read_AlignContainer(GeometryReader& reader) {
    std::unique_ptr<align::OneDirectionAligner<direction3D(alignDirection)>>
        aligner(align::fromStrUnique<direction3D(alignDirection)>(reader.source.requireAttribute("align")));
    shared_ptr< AlignContainer<dim, alignDirection> > result(new AlignContainer<dim, alignDirection>(*aligner));
    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children(reader,
        [&]() -> PathHints::Hint {
            return result->add(reader.readExactlyOneChild< typename AlignContainer<dim, alignDirection>::ChildType >(),
                                readPlace(reader, alignDirection));
        },
        [&]() {
            result->add(reader.readObject< typename AlignContainer<dim, alignDirection>::ChildType >());
        }
    );
    return result;
}

static GeometryReader::RegisterObjectReader align_container2Dtran_reader(AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::NAME,
                                                                         read_AlignContainer<2, Primitive<2>::DIRECTION_TRAN>);
static GeometryReader::RegisterObjectReader align_container2Dvert_reader(AlignContainer<2, Primitive<2>::DIRECTION_VERT>::NAME,
                                                                         read_AlignContainer<2, Primitive<2>::DIRECTION_VERT>);

static GeometryReader::RegisterObjectReader align_container3Dlong_reader(AlignContainer<3, Primitive<3>::DIRECTION_LONG>::NAME,
                                                                         read_AlignContainer<3, Primitive<3>::DIRECTION_LONG>);
static GeometryReader::RegisterObjectReader align_container3Dtran_reader(AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::NAME,
                                                                         read_AlignContainer<3, Primitive<3>::DIRECTION_TRAN>);
static GeometryReader::RegisterObjectReader align_container3Dvert_reader(AlignContainer<3, Primitive<3>::DIRECTION_VERT>::NAME,
                                                                         read_AlignContainer<3, Primitive<3>::DIRECTION_VERT>);


}   // namespace plask
