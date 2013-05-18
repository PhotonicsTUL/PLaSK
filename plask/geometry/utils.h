#ifndef PLASK__GEOMETRY_UTILS_H
#define PLASK__GEOMETRY_UTILS_H

#include "object.h"

/** @file
This file contains some utils usefull with geometry classes.
*/

namespace plask {

/**
 * Lazy cache of object bounding box.
 */
template <int dims>
struct BoundingBoxCache {

    typedef typename Primitive<dims>::Box BoundingBoxT;

private:

    /// Connection object with child. It is necessary since disconnectOnChileChanged doesn't work
    boost::signals2::connection connection_with_object;

    /// Cached bounding box
    BoundingBoxT boundingBox;

    const GeometryObjectD<dims>* object;

    bool isFresh; ///< @c true only if value in cache is fresh

    void connect() {
        if (this->object)
            connection_with_object = this->object.changed.at_front(boost::bind(&BoundingBoxCache<dims>::onObjectChanged, this, _1));
    }

public:

    /**
     * Get bouding box of connected object. Read it from cache if it's possible or update cache.
     * @return bouding box of connected object
     * @throw Exception if no object is connected
     */
    const BoundingBoxT& operator()() const {
        if (!isFresh) {
            //if (!object) return BoundingBoxT::invalidInstance();
            if (!object) throw Exception("BoundingBoxCache is not initialized or object was deleted, so can't get bounding box");
            boundingBox = object->getBoundingBox();
            isFresh = true;
        }
        return boundingBox;
    }

    /**
     * Refresh bounding box cache. Called by object changed signal.
     * @param evt
     */
    void onObjectChanged(const GeometryObject::Event& evt) {
        //if (evt.isResize())
        //    boundingBox = static_cast<GeometryObjectD<dims>&>(evt.source())->getBoundingBox();
        if (evt.isResize()) isFresh = false;
        if (evt.isDelete()) {
            object = nullptr;
            isFresh = false;
        }
    }

    /**
     * Set object for which bounding box should be cached.
     * @param object object for which bounding box should be cached, can be nullptr to disconnect
     */
    void setObject(const GeometryObjectD<dims>* object) {
        if (this->object == object) return;
        connection_with_object.disconnect();
        this->object = object;
        isFresh = false;
        connect();
    }

    /**
     * Set object for which bounding box should be cached.
     * @param object object for which bounding box should be cached
     */
    void setObject(const GeometryObjectD<dims>& object) {
        setObject(&object);
    }

    /**
     * Set object for which bounding box should be cached.
     * @param object object for which bounding box should be cached
     */
    void setObject(shared_ptr<const GeometryObjectD<dims> > object) {
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
    BoundingBoxCache(const GeometryObjectD<dims>* object = 0): object(object), isFresh(false) {
        connect();
    }

    /**
     * Construct bouding box cache connected with given object
     * @param object object for which bounding box should be cached
     */
    BoundingBoxCache(const GeometryObjectD<dims>& object): object(&object), isFresh(false) {
        connect();
    }

    /**
     * Construct bouding box cache connected with given object
     * @param object object for which bounding box should be cached
     */
    BoundingBoxCache(shared_ptr< const GeometryObjectD<dims> > object): object(&object), isFresh(false) {
        connect();
    }

    ~BoundingBoxCache() {
        connection_with_object.disconnect();
    }

};


}   // namespace plask

#endif // PLASK__GEOMETRY_UTILS_H
