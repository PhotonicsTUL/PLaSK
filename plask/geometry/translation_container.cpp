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

template <int DIMS>
struct EmptyLeafCacheNode: public CacheNode<DIMS> {

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const {
        return shared_ptr<Material>();
    }

    virtual bool includes(const Vec<DIMS>& p) const {
        return false;
    }

    GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &point, bool all) const {
        return GeometryObject::Subtree();
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

    virtual bool includes(const Vec<DIMS>& p) const {
        for (auto child: children) if (child->includes(p)) return true;
        return false;
    }

    GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &point, bool all) const {
        GeometryObject::Subtree result;
        for (auto child = children.rbegin(); child != children.rend(); ++child) {
            GeometryObject::Subtree child_path = (*child)->getPathsAt(point, all);
            if (!child_path.empty()) {
                result.children.push_back(std::move(child_path));
                if (!all) break;
            }
        }
        if (!result.children.empty())
            result.object = caller;
        return result;
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

    virtual bool includes(const Vec<DIMS>& p) const {
        return p[dir] < offset ? lo->includes(p) : hi->includes(p);
    }

    GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &p, bool all) const {
        return p[dir] < offset ? lo->getPathsAt(caller, p, all) : hi->getPathsAt(caller, p, all);
    }

    virtual ~InternalCacheNode() {
        delete lo;
        delete hi;
    }
};

/*template <int DIMS>
inline CacheNode<DIMS>* constructInternalNode(int dir, const double& offset, CacheNode<DIMS>* lo, CacheNode<DIMS>* hi) {
    static_assert(0, "DIMS must be 2 or 3");
}

template <>*/
inline CacheNode<2>* constructInternalNode(int dir, const double& offset, CacheNode<2>* lo, CacheNode<2>* hi) {
    if (dir == 0) return new InternalCacheNode<2, 0>(offset, lo, hi);
    assert(dir == 1);
    return new InternalCacheNode<2, 1>(offset, lo, hi);
}

//template <>
inline CacheNode<3>* constructInternalNode(int dir, const double& offset, CacheNode<3>* lo, CacheNode<3>* hi) {
    if (dir == 0) return new InternalCacheNode<3, 0>(offset, lo, hi);
    if (dir == 1) return new InternalCacheNode<3, 1>(offset, lo, hi);
    assert(dir == 2);
    return new InternalCacheNode<3, 2>(offset, lo, hi);
}


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

/**
 * Build cache.
 * @param input 1 + DIMS * 2 vectors sorted by:
 *  - 0 - in oryginal order,
 *  - dir*2 + 1 (for dim = 0 ... DIMS-1) - sorted by lower bound of bouding box in direction dir,
 *  - dir*2 + 2 (for dim = 0 ... DIMS-1) - sorted by upper bound of bouding box in direction dir.
 * Warning: this fynction change (destroy) inputs vectors.
 * @param max_depth maximum depth
 * @return constructed cache
 */
template <int DIMS>
inline CacheNode<DIMS>* buildCacheR(std::vector< GeometryObjectBBox<DIMS> >* input, int max_depth = 16) {
    if (input[0].size() < MIN_CHILD_TO_TRY_SPLIT || max_depth == 0) return new LeafCacheNode<DIMS>(input[0]);
    double bestOffset;
    int bestDir;
    int bestValue = std::numeric_limits<int>::max();  //we will minimalize this value
    for (int dim = DIMS; dim < DIMS; ++dim)
        calcOptimalSplitOffset(input[dim*2 + 1], input[dim*2 + 2], dim, bestDir, bestOffset, bestValue);
    if (bestValue == std::numeric_limits<int>::max())   //there are no enought good split point
        return new LeafCacheNode<DIMS>(input[0]);                //so we will not split more
    CacheNode<DIMS> *lo, *hi;
    {
    std::vector< GeometryObjectBBox<DIMS> > input_over_offset[1 + DIMS * 2];
    for (int dim = DIMS; dim < 1 + DIMS * 2; ++dim)
        inPlaceSplit<DIMS>(input[dim], input_over_offset[dim], bestDir, bestOffset);
    hi = buildCacheR(input_over_offset, max_depth-1);
    }   //here input_over_offset is deleted
    lo = buildCacheR(input, max_depth-1);
    return constructInternalNode(bestDir, bestOffset, lo, hi);
}

template <int DIMS>
CacheNode<DIMS>* buildCache(const typename GeometryObjectContainer<DIMS>::TranslationVector& children) {
    if (children.empty()) return new EmptyLeafCacheNode<DIMS>();
    if (children.size() < MIN_CHILD_TO_TRY_SPLIT) return new LeafCacheNode<DIMS>(children);
    std::vector< GeometryObjectBBox<DIMS> > input[1 + DIMS * 2];
    input[0].reserve(children.size());
    for (auto& c: children) input[0].emplace_back(c);
    for (int dir = 0; dir < DIMS; ++dir) {
        input[2*dir + 1] = input[0];
        std::sort(input[2*dir + 1].begin(), input[2*dir + 1].end(),
                [dir](const GeometryObjectBBox<DIMS>& a, const GeometryObjectBBox<DIMS>& b) {
                    return a.boundingBox.lower[dir] < b.boundingBox.lower[dir];
                }
        );
        input[2*dir + 2] = input[0];
        std::sort(input[2*dir + 2].begin(), input[2*dir + 2].end(),
                [dir](const GeometryObjectBBox<DIMS>& a, const GeometryObjectBBox<DIMS>& b) {
                    return a.boundingBox.upper[dir] < b.boundingBox.upper[dir];
                }
        );
    }
    return buildCacheR<DIMS>(input);
}


// ---- container: ----

template <int dim>
TranslationContainer<dim>::~TranslationContainer() {
    delete cache.load();
}

template <int dim>
PathHints::Hint TranslationContainer<dim>::addUnsafe(shared_ptr<TranslationContainer<dim>::ChildType> el, ChildAligner aligner) {
    invalidateCache();
    return this->_addUnsafe(newTranslation(el, aligner), aligner);
}

template <int dim>
PathHints::Hint TranslationContainer<dim>::addUnsafe(shared_ptr<TranslationContainer<dim>::ChildType> el, const TranslationContainer<dim>::DVec& translation) {
    /*shared_ptr<TranslationContainer<dim>::TranslationT> trans_geom(new TranslationContainer<dim>::TranslationT(el, translation));
    this->connectOnChildChanged(*trans_geom);
    children.push_back(trans_geom);
    invalidateCache();
    this->fireChildrenInserted(children.size()-1, children.size());
    return PathHints::Hint(shared_from_this(), trans_geom);*/
    return this->addUnsafe(el, align::fromVector(translation));
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
CacheNode<dim>* TranslationContainer<dim>::ensureHasCache() {
    if (!cache)
        cache = buildCache<dim>(children);
    return cache;
}

template <int dim>
CacheNode<dim>* TranslationContainer<dim>::ensureHasCache() const {
    if (cache) return cache;
    boost::lock_guard<boost::mutex> lock(const_cast<boost::mutex&>(cache_mutex));
    //this also will check if cache is non-null egain, someone could build cache when we waited for enter to critical section:
    return const_cast<TranslationContainer<dim>*>(this)->ensureHasCache();
}

template <int dim>
shared_ptr<GeometryObject> TranslationContainer<dim>::changedVersionForChildren(
        std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< TranslationContainer<dim> > result = make_shared< TranslationContainer<dim> >();
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first, children[child_no]->translation + vec<dim, double>(children_after_change[child_no].second));
    return result;
}

template <int dim>
shared_ptr<typename TranslationContainer<dim>::TranslationT> TranslationContainer<dim>::newTranslation(const shared_ptr<typename TranslationContainer<dim>::ChildType>& el, ChildAligner aligner) {
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el);
    aligner.align(*trans_geom);
    return trans_geom;
}

template struct TranslationContainer<2>;
template struct TranslationContainer<3>;

// ---- containers readers: ----

template <int dim>
shared_ptr<GeometryObject> read_TranslationContainer(GeometryReader& reader) {
    shared_ptr< TranslationContainer<dim> > result(new TranslationContainer<dim>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children(reader,
        [&]() -> PathHints::Hint {
            return result->add(
                        reader.readExactlyOneChild< typename TranslationContainer<dim>::ChildType >(),
                        align::fromXML(reader.source, *reader.axisNames, align::fromVector(Primitive<dim>::ZERO_VEC))
                   );
        },
        [&]() {
            result->add(reader.readObject< typename TranslationContainer<dim>::ChildType >());
        }
    );
    return result;
}

static GeometryReader::RegisterObjectReader container2D_reader(TranslationContainer<2>::NAME, read_TranslationContainer<2>);
static GeometryReader::RegisterObjectReader container3D_reader(TranslationContainer<3>::NAME, read_TranslationContainer<3>);

} // namespace plask
