#include "translation_container.h"

#include <cstdlib>  //abs

#define PLASK_TRANSLATIONCONTAINER2D_NAME ("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_TRANSLATIONCONTAINER3D_NAME ("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)

namespace plask {

template <int dim>
const char* TranslationContainer<dim>::NAME = dim == 2 ? PLASK_TRANSLATIONCONTAINER2D_NAME : PLASK_TRANSLATIONCONTAINER3D_NAME;

template <int dim>
TranslationContainer<dim>::~TranslationContainer() {
    delete cache.load();
}

template <int dim>
PathHints::Hint TranslationContainer<dim>::addUnsafe(shared_ptr<typename TranslationContainer<dim>::ChildType> el, ChildAligner aligner) {
    invalidateCache();
    return this->_addUnsafe(newTranslation(el, aligner), aligner);
}

template <int dim>
PathHints::Hint TranslationContainer<dim>::addUnsafe(shared_ptr<typename TranslationContainer<dim>::ChildType> el, const typename TranslationContainer<dim>::DVec& translation) {
    return this->addUnsafe(el, align::fromVector(translation));
}

template <int dim>
PathHints::Hint TranslationContainer<dim>::insertUnsafe(const std::size_t pos, shared_ptr<typename TranslationContainer<dim>::ChildType> el, ChildAligner aligner) {
    invalidateCache();
    return this->_insertUnsafe(pos, newTranslation(el, aligner), aligner);
}

template <int dim>
PathHints::Hint TranslationContainer<dim>::insertUnsafe(const std::size_t pos, shared_ptr<typename TranslationContainer<dim>::ChildType> el, const typename TranslationContainer<dim>::DVec& translation) {
    return this->insertUnsafe(pos, el, align::fromVector(translation));
}

/*template <>
void TranslationContainer<2>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    shared_ptr<Translation<2>> child_tran = children[child_index];
    if (child_tran->translation.tran() != 0.0) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.tran());
    if (child_tran->translation.vert() != 0.0) dest_xml_child_tag.attr(axes.getNameForVert(), child_tran->translation.vert());
}

template <>
void TranslationContainer<3>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    shared_ptr<Translation<3>> child_tran = children[child_index];
    if (child_tran->translation.lon() != 0.0) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.lon());
    if (child_tran->translation.tran() != 0.0) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.tran());
    if (child_tran->translation.vert() != 0.0) dest_xml_child_tag.attr(axes.getNameForVert(), child_tran->translation.vert());
}*/

template <int dim>
void TranslationContainer<dim>::invalidateCache() {
    delete cache.load();
    cache = nullptr;
}

template <int dim>
SpatialIndexNode<dim>* TranslationContainer<dim>::ensureHasCache() {
    if (!cache.load())
        cache = buildSpatialIndex<dim>(children).release();
    return cache;
}

template <int dim>
SpatialIndexNode<dim>* TranslationContainer<dim>::ensureHasCache() const {
    if (cache.load()) return cache;
    boost::lock_guard<boost::mutex> lock(const_cast<boost::mutex&>(cache_mutex));
    //this also will check if cache is non-null egain, someone could build cache when we waited for enter to critical section:
    return const_cast<TranslationContainer<dim>*>(this)->ensureHasCache();
}

template <int dim>
shared_ptr<GeometryObject> TranslationContainer<dim>::shallowCopy() const {
    shared_ptr<TranslationContainer<dim>> result = plask::make_shared<TranslationContainer<dim>>();
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        result->addUnsafe(children[child_no]->getChild(), children[child_no]->translation);
    return result;
}

template <int dim>
shared_ptr<GeometryObject> TranslationContainer<dim>::deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<TranslationContainer<dim>> result = plask::make_shared<TranslationContainer<dim>>();
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children[child_no]->getChild())
            result->addUnsafe(static_pointer_cast<ChildType>(children[child_no]->getChild()->deepCopy(copied)), children[child_no]->translation);
    return result;
    copied[this] = result;
    return result;
}


template <int dim>
shared_ptr<GeometryObject> TranslationContainer<dim>::changedVersionForChildren(
        std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* /*recomended_translation*/) const {
    shared_ptr<TranslationContainer<dim>> result = plask::make_shared<TranslationContainer<dim>>();
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first, children[child_no]->translation + vec<dim, double>(children_after_change[child_no].second));
    return result;
}

template <int dim>
shared_ptr<typename TranslationContainer<dim>::TranslationT> TranslationContainer<dim>::newTranslation(const shared_ptr<typename TranslationContainer<dim>::ChildType>& el, ChildAligner aligner) {
    shared_ptr<TranslationT> trans_geom = plask::make_shared<TranslationT>(el);
    aligner.align(*trans_geom);
    return trans_geom;
}

// ---- containers readers: ----

template <int dim>
shared_ptr<GeometryObject> read_TranslationContainer(GeometryReader& reader) {
    shared_ptr< TranslationContainer<dim> > result(new TranslationContainer<dim>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    auto default_aligner = align::fromXML(reader.source, reader.getAxisNames(), align::fromVector(Primitive<dim>::ZERO_VEC));
    read_children(reader,
        [&]() -> PathHints::Hint {
            auto aligner = align::fromXML(reader.source, reader.getAxisNames(), default_aligner);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<dim>::ChildType >(), aligner);
        },
        [&]() {
            result->add(reader.readObject< typename TranslationContainer<dim>::ChildType >(), default_aligner);
        }
    );
    return result;
}

static GeometryReader::RegisterObjectReader container2D_reader(PLASK_TRANSLATIONCONTAINER2D_NAME, read_TranslationContainer<2>);
static GeometryReader::RegisterObjectReader container3D_reader(PLASK_TRANSLATIONCONTAINER3D_NAME, read_TranslationContainer<3>);
static GeometryReader::RegisterObjectReader align_container2D_reader("align2d", read_TranslationContainer<2>);
static GeometryReader::RegisterObjectReader align_container3D_reader("align3d", read_TranslationContainer<3>);

template struct PLASK_API TranslationContainer<2>;
template struct PLASK_API TranslationContainer<3>;

} // namespace plask
