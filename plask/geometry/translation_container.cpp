#include "translation_container.h"

#include <cstdlib>  //abs

namespace plask {

// ---- cache: ----

/// Geometry object + his bounding box.
template <int DIMS>
struct GeometryObjectBBox {

    shared_ptr<const Translation<DIMS> > obj;

    typename Primitive<DIMS>::Box boundingBox;

    GeometryObjectBBox() {}

    GeometryObjectBBox(shared_ptr<const Translation<DIMS> > obj, const typename Primitive<DIMS>::Box& boundingBox)
        : obj(obj), boundingBox(boundingBox) {}

    GeometryObjectBBox(shared_ptr<const Translation<DIMS> > obj)
        : obj(obj), boundingBox(obj->getBoundingBox()) {}

};

template <int DIMS, int dir>
bool compare_by_lower(const GeometryObjectBBox<DIMS>& a, const GeometryObjectBBox<DIMS>& b) {
    return a.boundingBox.lower[dir] < b.boundingBox.lower[dir];
}

template <int DIMS, int dir>
bool compare_by_upper(const GeometryObjectBBox<DIMS>& a, const GeometryObjectBBox<DIMS>& b) {
    return a.boundingBox.upper[dir] < b.boundingBox.upper[dir];
}

template <int DIMS>
struct EmptyLeafCacheNode: public CacheNode<DIMS> {
    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const {
        return shared_ptr<Material>();
    }
};

template <int DIMS>
struct LeafCacheNode: public CacheNode<DIMS> {

    /// Type of the vector holding container children
    typedef std::vector< shared_ptr<const Translation<DIMS> > > ChildVectorT;

    ChildVectorT children;

    LeafCacheNode(const std::vector< GeometryObjectBBox<DIMS> >& children_with_bb) {
        children.reserve(children_with_bb.size());
        for (const GeometryObjectBBox<DIMS>& c: children_with_bb)
            children.push_back(c.obj);
    }

    LeafCacheNode(const std::vector< shared_ptr< Translation<DIMS> > >& childr) {
        children.reserve(childr.size());
        for (const shared_ptr< Translation<DIMS> >& c: childr)
            children.push_back(c);
    }

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const {
        for (auto child_it = children.rbegin(); child_it != children.rend(); ++child_it) {
            shared_ptr<Material> r = (*child_it)->getMaterial(p);
            if (r != nullptr) return r;
        }
        return shared_ptr<Material>();
    }
};

/// Instances of this template represents all internal nodes of cache
template <int DIMS, int dir>
struct InternalCacheNode: public CacheNode<DIMS> {

    double offset;  ///< split coordinate
    CacheNode<DIMS>* lo;  ///< includes all objects which has lower coordinate < offset
    CacheNode<DIMS>* hi;  ///< includes all objects which has higher coordinate >= offset

    InternalCacheNode(const double& offset, CacheNode<DIMS>* lo, CacheNode<DIMS>* hi)
        : offset(offset), lo(lo), hi(hi)
    {}

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const {
        return p[dir] < offset ? lo->getMaterial(p) : hi->getMaterial(p);
    }

    virtual ~InternalCacheNode() {
        delete lo;
        delete hi;
    }
};

template <int DIMS>
void inPlaceSplit(std::vector< GeometryObjectBBox<DIMS> >& inputAndLo, std::vector< GeometryObjectBBox<DIMS> >& hi, int dir, double offset) {
    std::vector< GeometryObjectBBox<DIMS> > lo;
    for (GeometryObjectBBox<DIMS>& i: inputAndLo) {
        if (i.boundingBox.lower[dir] < offset) lo.push_back(i);
        if (i.boundingBox.upper[dir] >= offset) hi.push_back(i);
    }
    std::swap(lo, inputAndLo);
}

/**
 * Calculate optimal spliting offset in given direction.
 * @param inputSortedByLo, inputSortedByHi input vector sorted by lo and hi boxes coordinates (in inputDir)
 * @param inputDir searched direction
 * @param bestDir, bestOffset, bestValue parameters of earlier best point, eventualy changed
 */
template <int DIMS>
void calcOptimalSplitOffset(const std::vector< GeometryObjectBBox<DIMS> >& inputSortedByLo, const std::vector< GeometryObjectBBox<DIMS> >& inputSortedByHi,
                            int inputDir, int& bestDir, double& bestOffset, int& bestValue)
{
    const int max_allowed_size = inputSortedByLo.size() - 4;
    std::size_t i_hi = 0;
    for (std::size_t i_lo = 1; i_lo < inputSortedByLo.size(); ++i_lo) {
        const double& offset = inputSortedByLo[i_lo].boundingBox.lower[inputDir];
        while (i_lo+1 < inputSortedByLo.size() && inputSortedByLo[i_lo+1].boundingBox.lower[inputDir] == offset)
            ++i_lo;   //can has more obj. with this lo coordinate
        //now: obj. from [0, i_lo) will be added to lo set
        if (i_lo > max_allowed_size)
            return; //too much obj in lo, i_lo will be increased so we can return
        while (i_hi < inputSortedByHi.size() && inputSortedByHi[i_hi].boundingBox.upper[inputDir] < offset)
            ++i_hi;
        //now: obj. from [i_hi, inputSortedByHi.size()) will be added to hi set
        const int hi_size = inputSortedByHi.size() - i_hi;
        if (hi_size > max_allowed_size)
            continue;   //too much obj in hi, we must wait for higher i_hi
        //common part is: [i_hi, i_lo)
        const int value = (i_lo - i_hi) * 3  //this is number of common obj in two sets * 3, we want to minimalize this
                   + std::abs(hi_size - i_lo);    //diffrent of set sizes, we also want to minimalize this
        if (value < bestValue) {
            bestValue = value;
            bestOffset = offset;
            bestDir = inputDir;
        }
    }
}

#define MIN_CHILD_TO_TRY_SPLIT 16

//warning: this destroy inputs vectors
CacheNode<2>* buildCache(std::vector< GeometryObjectBBox<2> >& input,
                         std::vector< GeometryObjectBBox<2> >& inputSortedByLoC0, std::vector< GeometryObjectBBox<2> >& inputSortedByHiC0,
                         std::vector< GeometryObjectBBox<2> >& inputSortedByLoC1, std::vector< GeometryObjectBBox<2> >& inputSortedByHiC1,
                         int max_depth = 16) {
    if (input.size() < MIN_CHILD_TO_TRY_SPLIT || max_depth == 0) return new LeafCacheNode<2>(input);
    double bestOffset;
    int bestDir;
    int bestValue = std::numeric_limits<int>::max();  //we will minimalize this value
    calcOptimalSplitOffset(inputSortedByLoC0, inputSortedByHiC0, 0, bestDir, bestOffset, bestValue);
    calcOptimalSplitOffset(inputSortedByLoC1, inputSortedByHiC1, 1, bestDir, bestOffset, bestValue);
    if (bestValue == std::numeric_limits<int>::max())   //there are no enought good split point
        return new LeafCacheNode<2>(input);                //so we will not split more
    CacheNode<2> *lo, *hi;
    {
    std::vector< GeometryObjectBBox<2> > input_over_offset,
            inputSortedByLoC0_over_offset, inputSortedByHiC0_over_offset,
            inputSortedByLoC1_over_offset, inputSortedByHiC1_over_offset;
    inPlaceSplit<2>(input, input_over_offset, bestDir, bestOffset);
    inPlaceSplit<2>(inputSortedByLoC0, inputSortedByLoC0_over_offset, bestDir, bestOffset);
    inPlaceSplit<2>(inputSortedByHiC0, inputSortedByHiC0_over_offset, bestDir, bestOffset);
    inPlaceSplit<2>(inputSortedByLoC1, inputSortedByLoC1_over_offset, bestDir, bestOffset);
    inPlaceSplit<2>(inputSortedByHiC1, inputSortedByHiC1_over_offset, bestDir, bestOffset);
    hi = buildCache(input_over_offset, inputSortedByLoC0_over_offset, inputSortedByHiC0_over_offset, inputSortedByLoC1_over_offset, inputSortedByHiC1_over_offset, max_depth-1);
    }   //here inputs over_offset are deleted
    lo = buildCache(input, inputSortedByLoC0, inputSortedByHiC0, inputSortedByLoC1, inputSortedByHiC1, max_depth-1);
    if (bestDir == 0) return new InternalCacheNode<2, 0>(bestOffset, lo, hi);
    assert(bestDir == 1);
    return new InternalCacheNode<2, 1>(bestOffset, lo, hi);
}

CacheNode<2>* buildCache(const GeometryObjectContainer<2>::TranslationVector& children) {
    if (children.empty()) return new EmptyLeafCacheNode<2>();
    if (children.size() < MIN_CHILD_TO_TRY_SPLIT) return new LeafCacheNode<2>(children);
    std::vector< GeometryObjectBBox<2> > input,
            inputSortedByLoC0, inputSortedByHiC0,
            inputSortedByLoC1, inputSortedByHiC1;
    input.reserve(children.size());
    for (auto& c: children) input.emplace_back(c);
    inputSortedByLoC0 = input;
    std::sort(inputSortedByLoC0.begin(), inputSortedByLoC0.end(), compare_by_lower<2, 0>);
    inputSortedByLoC1 = input;
    std::sort(inputSortedByLoC1.begin(), inputSortedByLoC1.end(), compare_by_lower<2, 1>);
    inputSortedByHiC0 = input;
    std::sort(inputSortedByHiC0.begin(), inputSortedByHiC0.end(), compare_by_upper<2, 0>);
    inputSortedByHiC1 = input;
    std::sort(inputSortedByHiC1.begin(), inputSortedByHiC1.end(), compare_by_upper<2, 1>);
    return buildCache(input,
                      inputSortedByLoC0, inputSortedByHiC0,
                      inputSortedByLoC1, inputSortedByHiC1);
}

CacheNode<3>* buildCache(const GeometryObjectContainer<3>::TranslationVector& children) {
    //TODO implementation simillar to 2D version (after testing this 2D version)
    return new LeafCacheNode<3>(children);
}


// ---- container: ----

template <>
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
}

template <int dim>
void TranslationContainer<dim>::invalidateCache() {
    delete cache;
    cache = nullptr;
}

template <int dim>
void TranslationContainer<dim>::ensureHasCache() {
    if (!cache)
        cache = buildCache(children);
}

template <int dim>
shared_ptr<GeometryObject> TranslationContainer<dim>::changedVersionForChildren(
        std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< TranslationContainer<dim> > result = make_shared< TranslationContainer<dim> >();
    for (std::size_t child_nr = 0; child_nr < children.size(); ++child_nr)
        if (children_after_change[child_nr].first)
            result->addUnsafe(children_after_change[child_nr].first, children[child_nr]->translation + vec<dim, double>(children_after_change[child_nr].second));
    return result;
}

template struct TranslationContainer<2>;
template struct TranslationContainer<3>;

// ---- containers readers: ----

shared_ptr<GeometryObject> read_TranslationContainer2D(GeometryReader& reader) {
    shared_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<TranslationContainer<2>>(reader,
        [&]() -> PathHints::Hint {
            TranslationContainer<2>::DVec translation;
            translation.tran() = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
            translation.vert() = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<2>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}

shared_ptr<GeometryObject> read_TranslationContainer3D(GeometryReader& reader) {
    shared_ptr< TranslationContainer<3> > result(new TranslationContainer<3>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children<TranslationContainer<3>>(reader,
        [&]() -> PathHints::Hint {
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

static GeometryReader::RegisterObjectReader container2D_reader(TranslationContainer<2>::NAME, read_TranslationContainer2D);
static GeometryReader::RegisterObjectReader container3D_reader(TranslationContainer<3>::NAME, read_TranslationContainer3D);

} // namespace plask
