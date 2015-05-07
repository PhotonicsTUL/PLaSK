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
struct SpetialIndexNode {

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const = 0;

    virtual bool contains(const Vec<DIMS>& p) const = 0;

    virtual GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &point, bool all) const = 0;

    virtual ~SpetialIndexNode() {}
};


/**
 * Build spetial index.
 *
 * The index prevent reverse order of children in case of material searching.
 * @param children vector of geometry object for which index should be build
 * @return constructed index
 */
template <int DIMS>
std::unique_ptr<SpetialIndexNode<DIMS>> buildSpetialIndex(const std::vector< shared_ptr<Translation<DIMS>> >& children);

PLASK_API_EXTERN_TEMPLATE_STRUCT(SpetialIndexNode<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(SpetialIndexNode<3>)

extern template PLASK_API std::unique_ptr<SpetialIndexNode<2>> buildSpetialIndex(const std::vector< shared_ptr<Translation<2>> >& children);
extern template PLASK_API std::unique_ptr<SpetialIndexNode<3>> buildSpetialIndex(const std::vector< shared_ptr<Translation<3>> >& children);

}   // plask

#endif // PLASK__GEOMETRY_SPATIAL_INDEX_H
