#ifndef PLASK__GEOMETRY_UTILS_H
#define PLASK__GEOMETRY_UTILS_H

#include "object.h"

/** @file
This file includes some utils usefull with geometry classes.
*/

namespace plask {

/**
 * Cache of object bounding box.
 */
template <int dims>
struct BoundingBoxCache {

    /// Connection object with child. It is necessary since disconnectOnChileChanged doesn't work
    boost::signals2::connection connection_with_object;

    /// Cached bounding box
    typename Primitive<dims>::Box boundingBox;

    /**
     * Refresh bounding box cache. Called by object changed signal.
     * @param evt
     */
    void onObjectChanged(const GeometryObject::Event& evt) {
        if (evt.isResize())
            boundingBox = static_cast<GeometryObjectD<dims>&>(evt.source())->getBoundingBox();
    }

    /**
     * Initialize bounding box cache.
     */
    void setObject(GeometryObjectD<dims>& object) {
        connection_with_object.disconnect();
        connection_with_object = object.changed.at_front(boost::bind(&BoundingBoxCache<dims>::onObjectChanged, this, _1));
        boundingBox = object.getBoundingBox();
    }

    void setObject(shared_ptr< GeometryObjectD<dims> > object) {
        setObject(*object);
    }

    BoundingBoxCache(GeometryObjectD<dims>& object) {
        setObject(object);
    }

    BoundingBoxCache(shared_ptr< GeometryObjectD<dims> > object) {
        setObject(object);
    }

    ~BoundingBoxCache() {
        connection_with_object.disconnect();
    }

};


}   // namespace plask

#endif // PLASK__GEOMETRY_UTILS_H
