#ifndef PLASK__GEOMETRY_UTILS_H
#define PLASK__GEOMETRY_UTILS_H

#include "element.h"

/** @file
This file includes some utils usefull with geometry classes.
*/

namespace plask {

/**
 * Cache of element bounding box.
 */
template <int dims>
struct BoundingBoxCache {

    /// Connection object with child. It is necessary since disconnectOnChileChanged doesn't work
    boost::signals2::connection connection_with_element;

    /// Cached bounding box
    typename Primitive<dims>::Box boundingBox;

    /**
     * Refresh bounding box cache. Called by element changed signal.
     * @param evt
     */
    void onElementChanged(const GeometryElement::Event& evt) {
        if (evt.isResize())
            boundingBox = static_cast<GeometryElementD<dims>&>(evt.source())->getBoundingBox();
    }

    /**
     * Initialize bounding box cache.
     */
    void setElement(GeometryElementD<dims>& element) {
        connection_with_element.disconnect();
        connection_with_element = element.changed.at_front(boost::bind(&BoundingBoxCache<dims>::onElementChanged, this, _1));
        boundingBox = element.getBoundingBox();
    }

    void setElement(shared_ptr< GeometryElementD<dims> > element) {
        setElement(*element);
    }

    BoundingBoxCache(GeometryElementD<dims>& element) {
        setElement(element);
    }

    BoundingBoxCache(shared_ptr< GeometryElementD<dims> > element) {
        setElement(element);
    }

    ~BoundingBoxCache() {
        connection_with_element.disconnect();
    }

};


}   // namespace plask

#endif // PLASK__GEOMETRY_UTILS_H
