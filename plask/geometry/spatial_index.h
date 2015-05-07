#ifndef PLASK__GEOMETRY_SPATIAL_INDEX_H
#define PLASK__GEOMETRY_SPATIAL_INDEX_H

#include "transform.h"

namespace plask {

/**
 * Base class for cache and nodes of cache.
 *
 * It has some methods similar to this in GeometryObjectContainer API and is used by TranslationContainer.
 */
template <int DIMS>
struct CacheNode {

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const = 0;

    virtual bool contains(const Vec<DIMS>& p) const = 0;

    virtual GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &point, bool all) const = 0;

    virtual ~CacheNode() {}
};



template <int DIMS>
CacheNode<DIMS>* buildCache(const std::vector< shared_ptr<Translation<DIMS>> >& children);

PLASK_API_EXTERN_TEMPLATE_STRUCT(CacheNode<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(CacheNode<3>)

extern template PLASK_API CacheNode<2>* buildCache(const std::vector< shared_ptr<Translation<2>> >& children);
extern template PLASK_API CacheNode<3>* buildCache(const std::vector< shared_ptr<Translation<3>> >& children);

}   // plask

#endif // PLASK__GEOMETRY_SPATIAL_INDEX_H
