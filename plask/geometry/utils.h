#ifndef PLASK__GEOMETRY_UTILS_H
#define PLASK__GEOMETRY_UTILS_H

#include "object.h"

/** @file
This file contains some utils usefull with geometry classes.
*/

namespace plask {

//TODO at the moment, this template is not used, consider to remove it
/**
 * Lazy cache of object bounding box.
 */
template <int dims>
struct PLASK_API BoundingBoxCache {

    typedef typename Primitive<dims>::Box BoundingBoxT;

private:

    /// Connection object with child. It is necessary since disconnectOnChileChanged doesn't work
    boost::signals2::connection connection_with_object;

    /// Cached bounding box
    BoundingBoxT boundingBox;

    GeometryObjectD<dims>* object;

    bool isFresh; ///< @c true only if value in cache is fresh

    void connect();

public:

    /**
     * Get bouding box of connected object. Read it from cache if it's possible or update cache.
     * @return bouding box of connected object
     * @throw Exception if no object is connected
     */
    const BoundingBoxT& operator()();

    /**
     * Refresh bounding box cache. Called by object changed signal.
     * @param evt
     */
    void onObjectChanged(const GeometryObject::Event& evt);

    /**
     * Set object for which bounding box should be cached.
     * @param object object for which bounding box should be cached, can be nullptr to disconnect
     */
    void setObject(GeometryObjectD<dims>* object);

    /**
     * Set object for which bounding box should be cached.
     * @param object object for which bounding box should be cached
     */
    void setObject(GeometryObjectD<dims>& object) {
        setObject(&object);
    }

    /**
     * Set object for which bounding box should be cached.
     * @param object object for which bounding box should be cached
     */
    void setObject(shared_ptr<GeometryObjectD<dims> > object) {
        setObject(*object);
    }

    /**
     * Get object for which bounding box is cached.
     * @return object for which bounding box is cached or @c nullptr if bounding box is not cached, for any object
     */
    GeometryObjectD<dims>* getObject() const {
        return object;
    }

    /**
     * Construct bouding box cache connected with given object
     * @param object object for which bounding box should be cached, can be nullptr (typically you should later call setObject in such cache)
     */
    BoundingBoxCache(GeometryObjectD<dims>* object = 0): object(object), isFresh(false) {
        connect();
    }

    /**
     * Construct bouding box cache connected with given object
     * @param object object for which bounding box should be cached
     */
    BoundingBoxCache(GeometryObjectD<dims>& object): object(&object), isFresh(false) {
        connect();
    }

    /**
     * Construct bouding box cache connected with given object
     * @param object object for which bounding box should be cached
     */
    BoundingBoxCache(shared_ptr< GeometryObjectD<dims> > object): object(object.get()), isFresh(false) {
        connect();
    }

    ~BoundingBoxCache() {
        connection_with_object.disconnect();
    }

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(BoundingBoxCache<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(BoundingBoxCache<3>)

}   // namespace plask

#endif // PLASK__GEOMETRY_UTILS_H
