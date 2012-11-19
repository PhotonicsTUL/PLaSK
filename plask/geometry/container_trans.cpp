#include "container_trans.h"

namespace plask {

// ---- cache: ----

template <int DIMS>
struct EmptyLeafCacheNode: public CacheNode<DIMS> {
    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const {
        return shared_ptr<Material>();
    }
};

template <int DIMS>
struct LeafCacheNode: public CacheNode<DIMS> {
    
    /// Type of the vector holding container children
    typedef std::vector< shared_ptr<Translation<DIMS> > > ChildVectorT;

    ChildVectorT children;
    
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
    CacheNode<DIMS>* lo;  ///< includes all objects which has lower coordinate <= offset
    CacheNode<DIMS>* hi;  ///< includes all objects which has higher coordinate > offset
        
    InternalCacheNode(const double& offset, CacheNode<DIMS>* lo, CacheNode<DIMS>* hi)
        : offset(offset), lo(lo), hi(hi)
    {}
    
    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const {
        return p[dir] <= offset ? lo->getMaterial(p) : hi->getMaterial(p);
    }
    
    virtual ~InternalCacheNode() {
        delete lo;
        delete hi;
    }
};

/// Geometry object + his bounding box.
template <int DIMS>
struct WithBB {
    
    shared_ptr<const Translation<DIMS> > obj;

    typename Primitive<DIMS>::Box boundingBox;
    
    WithBB() {}
    
    WithBB(shared_ptr<const Translation<DIMS> > obj, const typename Primitive<DIMS>::Box& boundingBox)
        : obj(obj), boundingBox(boundingBox) {}
    
    WithBB(shared_ptr<const Translation<DIMS> > obj)
        : obj(obj), boundingBox(obj->getBoundingBox()) {}
    
};

template <int DIMS>
void inPlaceSplit(std::vector< const WithBB<DIMS> >& inputAndLo, std::vector< const WithBB<DIMS> >& hi, int dir, double offset) {
    std::vector< const WithBB<DIMS> > lo;
    for (const WithBB<DIMS>& i: inputAndLo) {
        if (i.boundingBox.lower[dir] <= offset) lo.push_back(i);
        if (i.boundingBox.higher[dir] > offset) hi.push_back(i);
    }
    std::swap(lo, inputAndLo);
}

template <int DIMS>
void calcOptimalSplitOffset(const std::vector< const WithBB<DIMS> >& inputSortedByLo, const std::vector< const WithBB<DIMS> >& inputSortedByHi,
                            int inputDir, int& bestDir, double& bestOffset, int& bestCommon, int& bestDiff)
{
    
}

CacheNode<2>* buildCache2D(std::vector< WithBB<2> > input,
                           std::vector< WithBB<2> > inputSortedByLoC0, std::vector< WithBB<2> > inputSortedByHiC0,
                           std::vector< WithBB<2> > inputSortedByLoC1, std::vector< WithBB<2> > inputSortedByHiC1) {
    //if (input.size() < 8)
    double bestOffset;
    int bestDir;
    int bestCommon;
    int bestDiff;
    //calcOptimalSplitOffset
    //calcOptimalSplitOffset
    //if (bestDiff < 2)
    //inPlaceSplit
    //CacheNode<2>* lo = buildCache2D()
    //CacheNode<2>* hi = buildCache2D()
    //if (bestDir == 0) return new InternalCacheNode<2, 0>(bestOffset, lo, hi);
    assert(bestDir == 1);
    //return new InternalCacheNode<2, 1>(bestOffset, lo, hi);
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
