#include "spatial_index.h"

namespace plask {


/// Geometry object + its bounding box.
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
struct EmptyLeafCacheNode: public SpatialIndexNode<DIMS> {

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& /*p*/) const override {
        return shared_ptr<Material>();
    }

    virtual bool contains(const Vec<DIMS>& /*p*/) const override {
        return false;
    }

    GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> /*caller*/, const Vec<DIMS> &/*point*/, bool /*all*/) const override {
        return GeometryObject::Subtree();
    }
};

template <int DIMS>
struct LeafCacheNode: public SpatialIndexNode<DIMS> {

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

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const override {
        for (auto child_it = children.rbegin(); child_it != children.rend(); ++child_it) {
            shared_ptr<Material> r = (*child_it)->getMaterial(p);
            if (r != nullptr) return r;
        }
        return shared_ptr<Material>();
    }

    virtual bool contains(const Vec<DIMS>& p) const override {
        for (auto child: children) if (child->contains(p)) return true;
        return false;
    }

    GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &point, bool all) const override {
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
        /*GeometryObject::Subtree result;   //general container version:
        if (all) {
            for (auto child = children.begin(); child != children.end(); ++child) {
                GeometryObject::Subtree child_path = (*child)->getPathsAt(point, true);
                if (!child_path.empty())
                    result.children.push_back(std::move(child_path));
            }
        } else {
            for (auto child = children.rbegin(); child != children.rend(); ++child) {
                GeometryObject::Subtree child_path = (*child)->getPathsAt(point, false);
                if (!child_path.empty()) {
                    result.children.push_back(std::move(child_path));
                    break;
                }
            }
        }
        if (!result.children.empty())
            result.object = caller;
        return result;*/
    }
};

/// Instances of this template represents all internal nodes of cache
template <int DIMS, int dir>
struct InternalCacheNode: public SpatialIndexNode<DIMS> {

    double offset;  ///< split coordinate
    SpatialIndexNode<DIMS>* lo;  ///< contains all objects which has lower coordinate < offset
    SpatialIndexNode<DIMS>* hi;  ///< contains all objects which has higher coordinate >= offset

    InternalCacheNode(const double& offset, SpatialIndexNode<DIMS>* lo, SpatialIndexNode<DIMS>* hi)
        : offset(offset), lo(lo), hi(hi)
    {}

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const override {
        return p[dir] < offset ? lo->getMaterial(p) : hi->getMaterial(p);
    }

    virtual bool contains(const Vec<DIMS>& p) const override {
        return p[dir] < offset ? lo->contains(p) : hi->contains(p);
    }

    GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &p, bool all) const override {
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
inline SpatialIndexNode<2>* constructInternalNode(int dir, const double& offset, SpatialIndexNode<2>* lo, SpatialIndexNode<2>* hi) {
    if (dir == 0) return new InternalCacheNode<2, 0>(offset, lo, hi);
    assert(dir == 1);
    return new InternalCacheNode<2, 1>(offset, lo, hi);
}

//template <>
inline SpatialIndexNode<3>* constructInternalNode(int dir, const double& offset, SpatialIndexNode<3>* lo, SpatialIndexNode<3>* hi) {
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
    const int max_allowed_size = int(inputSortedByLo.size()) - 4;
    std::size_t i_hi = 0;
    for (std::size_t i_lo = 1; i_lo < inputSortedByLo.size(); ++i_lo) {
        const double& offset = inputSortedByLo[i_lo].boundingBox.lower[inputDir];
        while (i_lo+1 < inputSortedByLo.size() && inputSortedByLo[i_lo+1].boundingBox.lower[inputDir] == offset)
            ++i_lo;   //can has more obj. with this lo coordinate
        //now: obj. from [0, i_lo) will be added to lo set
        if (int(i_lo) > max_allowed_size)
            return; //too much obj in lo, i_lo will be increased so we can return
        while (i_hi < inputSortedByHi.size() && inputSortedByHi[i_hi].boundingBox.upper[inputDir] < offset)
            ++i_hi;
        //now: obj. from [i_hi, inputSortedByHi.size()) will be added to hi set
        const int hi_size = int(inputSortedByHi.size()) - int(i_hi);
        if (hi_size > max_allowed_size)
            continue;   //too much obj in hi, we must wait for higher i_hi
        //common part is: [i_hi, i_lo)
        const int value = (int(i_lo) - int(i_hi)) * 3  //this is number of common obj in two sets * 3, we want to minimalize this
                   + std::abs(int(hi_size) - int(i_lo));    //diffrent of set sizes, we also want to minimalize this
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
 *  - 0 - in original order,
 *  - dir*2 + 1 (for dim = 0 ... DIMS-1) - sorted by lower bound of bouding box in direction dir,
 *  - dir*2 + 2 (for dim = 0 ... DIMS-1) - sorted by upper bound of bouding box in direction dir.
 * Warning: this fynction change (destroy) inputs vectors.
 * @param max_depth maximum depth
 * @return constructed cache
 */
template <int DIMS>
inline SpatialIndexNode<DIMS>* buildCacheR(std::vector< GeometryObjectBBox<DIMS> >* input, int max_depth = 16) {
    if (input[0].size() < MIN_CHILD_TO_TRY_SPLIT || max_depth == 0) return new LeafCacheNode<DIMS>(input[0]);
    double bestOffset;
    int bestDir;
    int bestValue = std::numeric_limits<int>::max();  //we will minimalize this value
    for (int dim = DIMS; dim < DIMS; ++dim)
        calcOptimalSplitOffset(input[dim*2 + 1], input[dim*2 + 2], dim, bestDir, bestOffset, bestValue);
    if (bestValue == std::numeric_limits<int>::max())   //there are no enought good split point
        return new LeafCacheNode<DIMS>(input[0]);                //so we will not split more
    SpatialIndexNode<DIMS> *lo, *hi;
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
std::unique_ptr<SpatialIndexNode<DIMS>> buildSpatialIndex(const std::vector< shared_ptr<Translation<DIMS>> >& children) {
    if (children.empty()) return std::unique_ptr<SpatialIndexNode<DIMS>>(new EmptyLeafCacheNode<DIMS>());
    if (children.size() < MIN_CHILD_TO_TRY_SPLIT) return std::unique_ptr<SpatialIndexNode<DIMS>>(new LeafCacheNode<DIMS>(children));
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
    return std::unique_ptr<SpatialIndexNode<DIMS>>(buildCacheR<DIMS>(input));
}

template struct PLASK_API SpatialIndexNode<2>;
template struct PLASK_API SpatialIndexNode<3>;

template PLASK_API std::unique_ptr<SpatialIndexNode<2>> buildSpatialIndex(const std::vector< shared_ptr<Translation<2>> >& children);
template PLASK_API std::unique_ptr<SpatialIndexNode<3>> buildSpatialIndex(const std::vector< shared_ptr<Translation<3>> >& children);


}   // namespace plask

