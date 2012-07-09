#ifndef PLASK__UTILS_CACHE_H
#define PLASK__UTILS_CACHE_H

/** @file
This file includes cache map indexed by objects which can emit events, like GeometryElements, Meshes, etc.
*/

#include <map>
#include "../memory.h"

namespace plask {

template <typename Key, typename Value>
struct CacheRemoveStrategyBase {

    /// Cached elements.
    std::map< Key*, plask::weak_ptr<Value> > map;

};

template <typename Key, typename Value>
struct CacheRemoveOnlyWhenDelete: public CacheRemoveStrategyBase<Key, Value> {

    /// If evt is delete event, remove source of event from cache map.
    void onEvent(typename Key::Event& evt) {
        if (evt.isDelete()) this->map.erase(&evt.source());
    }

};

template <typename Key, typename Value>
struct CacheRemoveOnEachChange: public CacheRemoveStrategyBase<Key, Value> {

    /// Remove source of event from cache map.
    void onEvent(typename Key::Event& evt) {
        auto& src = evt.source();
        src.changedDisconnectMethod(this, &CacheRemoveOnEachChange::onEvent);
        this->map.erase(&src);
    }

};

/**
 * Cache values of type Value using Key type to index it.
 * @tparam Key type using as index in cache (pointer to this type will be used), must be able to emit events;
 * @tparam Value type for cache values, will be stored in weak_ptr;
 * @tparam deleteStrategy when cache entries should be delete:
 * - CacheRemoveOnlyWhenDelete - when key is deleted (default),
 * - CacheRemoveOnEachChange - when key is changed,
 * - other class template which derive from plask::CacheRemoveStrategyBase and have void onEvent(typename Key::Event& evt) method - custom.
 */
template <typename Key, typename Value, template<typename Key, typename Value> class DeleteStrategy = CacheRemoveOnlyWhenDelete >
struct Cache: public DeleteStrategy<Key, Value> {

    /// Clear cache.
    ~Cache() {
        clear();
    }

    /**
     * Try get element from cache.
     *
     * Try also clean entry with @p el index if value for it is not still valid.
     * @param index key of element
     * @return non-null value from cache stored for key or null_ptr if there is no value for given index or value was not valid
     */
    plask::shared_ptr<Value> get(Key* index) {
        auto iter = this->map.find(index);
        if (iter != this->map.end()) {
            if (auto res = iter->second.lock())
                return res;
            else {
                iter->first->changedDisconnectMethod(this, &DeleteStrategy<Key, Value>::onEvent);
                this->map.erase(iter);
            }
        }
        return plask::shared_ptr<Value>();
    }

    /**
     * Try get element from cache.
     *
     * Try also clean entry with @p el index if value for it is not still valid.
     * @param index key of element
     * @return non-null value from cache stored for key or null_ptr if there is no value for given index or value was not valid
     */
    plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) {
        return get(index.get());
    }

    /**
     * Try get element from cache.
     * @param index key of element
     * @return non-null value from cache stored for key or null_ptr if there is no value for given index or value is not valid
     */
    plask::shared_ptr<Value> get(Key* index) const {
        auto constr_iter = this->map.find(index);
        if (constr_iter != this->map.end()) {
            if (auto res = constr_iter->second.lock())
                return res;
        }
        return plask::shared_ptr<Value>();
    }

    /**
     * Try get element from cache.
     * @param index key of element
     * @return non-null value from cache stored for key or null_ptr if there is no value for given index or value is not valid
     */
    plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) const {
        return get(index.get());
    }

    /**
     * Append entry to cache.
     * @param index key of entry
     * @param value value of entry
     */
    void append(Key* index, weak_ptr<Value> value) {
        this->map[index] = value;
        index->changedConnectMethod(this, &DeleteStrategy<Key, Value>::onEvent);
    }

    /**
     * Append entry to cache.
     * @param index key of entry
     * @param value value of entry
     */
    void append(plask::shared_ptr<Key> index, weak_ptr<Value> value) {
        append(index.get(), value);
    }

    /**
     * Construct shared pointer to value and append cache entry which consist of given index and constructed shared pointer.
     *
     * This is usefull in methods which wants to append new value to cache and return it:
     * @code
     * plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) {
     *   if (auto res = my_cache.get(index))
     *      return res;
     *   else
     *      return my_cache(index.get(), calculate_value_for(index));
     * }
     * @endcode
     * @param index, value entry data
     * @return shared pointer to value
     */
    plask::shared_ptr<Value> operator()(Key* index, Value* value) {
        plask::shared_ptr<Value> result(value);
        append(index, result);
        return result;
    }

    /**
     * Append cache entry which consist of given index and value and return value.
     *
     * This is usefull in methods which wants to append new value to cache and return it:
     * @code
     * plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) {
     *   if (auto res = my_cache.get(index))
     *      return res;
     *   else
     *      return my_cache(index.get(), calculate_value_for(index));
     * }
     * @endcode
     * @param index, value entry data
     * @return shared pointer to value
     */
    plask::shared_ptr<Value> operator()(Key* index, shared_ptr<Value> value) {
        append(index, value);
        return value;
    }


    /**
     * Construct shared pointer to value and append cache entry which consist of given index and constructed shared pointer.
     *
     * This is usefull in methods which wants to append new value to cache and return it:
     * @code
     * plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) {
     *   if (auto res = my_cache.get(index))
     *      return res;
     *   else
     *      return my_cache(index, calculate_value_for(index));
     * }
     * @endcode
     * @param index, value entry data
     * @return value
     */
    plask::shared_ptr<Value> operator()(plask::shared_ptr<Key> index, Value* value) {
        plask::shared_ptr<Value> result(value);
        append(index, result);
        return result;
    }

    /**
     * Construct shared pointer to value and append cache entry which consist of given index and constructed shared pointer.
     *
     * This is usefull in methods which wants to append new value to cache and return it:
     * @code
     * plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) {
     *   if (auto res = my_cache.get(index))
     *      return res;
     *   else
     *      return my_cache(index, calculate_value_for(index));
     * }
     * @endcode
     * @param index, value entry data
     * @return value
     */
    plask::shared_ptr<Value> operator()(plask::shared_ptr<Key> index, shared_ptr<Value> value) {
        append(index, value);
        return value;
    }

    //TODO operator() variants Value constructor parameters, ...?

    /**
     * Clean all entries for which values is already deleted.
     */
    void cleanDeleted() {
        for(auto i = this->map.begin(); i != this->map.end(); )
            if (i->second.expired()) {
                i->first.changedDisconnectMethod(this, &DeleteStrategy<Key, Value>::onEvent);
                this->map.erase(i++);
            }
                else ++i;
    }

    /**
     * Remove all entries from this cache.
     */
    void clear() {
        for (auto i: this->map)
            i.first->changedDisconnectMethod(this, &DeleteStrategy<Key, Value>::onEvent);
        this->map.clear();
    }
};

}   // namespace plask


#endif // CACHE_H
