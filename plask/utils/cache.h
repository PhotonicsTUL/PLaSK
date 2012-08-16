#ifndef PLASK__UTILS_CACHE_H
#define PLASK__UTILS_CACHE_H

/** @file
This file includes cache map indexed by objects which can emit events, like GeometryElements, Meshes, etc.
*/

#include <map>
#include "../memory.h"

namespace plask {

/**
 * Base class for strategy of removing from cache.
 *
 * Sublass should has method:
 * <code>void onEvent(typename Key::Event& evt)</code>
 * which eventualy (depends from event details) removes from map a source of event.
 *
 * Used by @ref WeakCache and @ref StrongCache.
 */
template <typename Key, typename ValuePtr>
struct CacheRemoveStrategyBase {

    /// Cached elements.
    std::map<Key*, ValuePtr> map;

};

/**
 * Strategy of removing from cache which removes key only when it is deleted.
 *
 * Used by @ref WeakCache and @ref StrongCache.
 */
template <typename Key, typename ValuePtr>
struct CacheRemoveOnlyWhenDeleted: public CacheRemoveStrategyBase<Key, ValuePtr> {

    /// If evt is delete event, remove source of event from cache map.
    void onEvent(typename Key::Event& evt) {
        if (evt.isDelete()) this->map.erase(&evt.source());
    }

};

/**
 * Strategy of removing from cache which removes key always when it is changed.
 *
 * Used by @ref WeakCache and @ref StrongCache.
 */
template <typename Key, typename ValuePtr>
struct CacheRemoveOnEachChange: public CacheRemoveStrategyBase<Key, ValuePtr> {

    /// Remove source of event from cache map.
    void onEvent(typename Key::Event& evt) {
        auto& src = evt.source();
        src.changedDisconnectMethod(this, &CacheRemoveOnEachChange::onEvent);
        this->map.erase(&src);
    }

};

template <typename Key, typename ValuePtr, template<typename SKey, typename SValuePtr> class DeleteStrategy = CacheRemoveOnlyWhenDeleted >
struct CacheBase: public DeleteStrategy<Key, ValuePtr> {

    typedef typename ValuePtr::element_type Value;

    /// Clear cache.
    ~CacheBase() {
        clear();
    }

    /**
     * Append entry to cache.
     * @param index key of entry
     * @param value value of entry
     */
    void append(Key* index, ValuePtr value) {
        this->map[index] = value;
       // index->changed.at_front(boost::bind(method, obj, _1));
        index->changedConnectMethod(this, &DeleteStrategy<Key, ValuePtr>::onEvent, boost::signals2::at_front);
    }

    /**
     * Append entry to cache.
     * @param index key of entry
     * @param value value of entry
     */
    void append(plask::shared_ptr<Key> index, ValuePtr value) {
        append(index.get(), value);
    }

    /**
     * Construct shared pointer to value and append cache entry which consists of given index and constructed shared pointer.
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
        this->append(index, result);
        return result;
    }

    /**
     * Append cache entry which consists of given index and value and return value.
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
        this->append(index, value);
        return value;
    }

    /**
     * Construct shared pointer to value and append cache entry which consists of given index and constructed shared pointer.
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
        this->append(index, result);
        return result;
    }

    /**
     * Construct shared pointer to value and append cache entry which consists of given index and constructed shared pointer.
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
        this->append(index, value);
        return value;
    }

    /**
     * Remove all entries from this cache.
     */
    void clear() {
        for (auto i: this->map)
            i.first->changedDisconnectMethod(this, &DeleteStrategy<Key, ValuePtr>::onEvent);
        this->map.clear();
    }

};

/**
 * Cache values of type Value using Key type to index it.
 *
 * It sores only weak_ptr to values, so it not prevent values from deletion.
 * Cache entires are removed on key changes (see @p deleteStrategy) or when value expires (only at moment of getting from non-const cache or calling cleanDeleted).
 * @tparam Key type using as index in cache (pointer to this type will be used), must be able to emit events;
 * @tparam Value type for cache values, will be stored in weak_ptr;
 * @tparam deleteStrategy when cache entries should be deleted:
 * - CacheRemoveOnlyWhenDeleted - when key is deleted (default),
 * - CacheRemoveOnEachChange - when key is changed,
 * - other class template which derives from plask::CacheRemoveStrategyBase and have void onEvent(typename Key::Event& evt) method - custom.
 */
template <typename Key, typename Value, template<typename SKey, typename SValuePtr> class DeleteStrategy = CacheRemoveOnlyWhenDeleted >
struct WeakCache: public CacheBase<Key, plask::weak_ptr<Value>, DeleteStrategy> {

    /**
     * Try get element from cache.
     *
     * Try also clean entry with @p el index if value for it is not still valid.
     * @param index key of element
     * @return non-null value from cache stored for key or nullptr if there is no value for given index or value was not valid
     */
    plask::shared_ptr<Value> get(Key* index) {
        auto iter = this->map.find(index);
        if (iter != this->map.end()) {
            if (auto res = iter->second.lock())
                return res;
            else {
                iter->first->changedDisconnectMethod(this, &DeleteStrategy<Key, plask::weak_ptr<Value> >::onEvent);
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
     * @return non-null value from cache stored for key or nullptr if there is no value for given index or value was not valid
     */
    plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) {
        return get(index.get());
    }

    /**
     * Try get element from cache.
     * @param index key of element
     * @return non-null value from cache stored for key or nullptr if there is no value for given index or value is not valid
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
     * @return non-null value from cache stored for key or nullptr if there is no value for given index or value is not valid
     */
    plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) const {
        return get(index.get());
    }

    /**
     * Clean all entries for which values are already deleted.
     *
     * This method has O(N+DlogN) time complexity, where N is number of elements in cache, and D is number of deleted elements.
     */
    void cleanDeleted() {
        for(auto i = this->map.begin(); i != this->map.end(); )
            if (i->second.expired()) {
                i->first.changedDisconnectMethod(this, &DeleteStrategy<Key, plask::weak_ptr<Value> >::onEvent);
                this->map.erase(i++);
            }
                else ++i;
    }

};

/**
 * Cache values of type Value using Key type to index it.
 *
 * It stores shared_ptr to values, so each value will be live when it's only is included in cache.
 * Cache entires are removed on key changes (see @p deleteStrategy).
 * @tparam Key type using as index in cache (pointer to this type will be used), must be able to emit events;
 * @tparam Value type for cache values, will be stored in shared_ptr;
 * @tparam deleteStrategy when cache entries should be deleted:
 * - CacheRemoveOnlyWhenDeleted - when key is deleted (default),
 * - CacheRemoveOnEachChange - when key is changed,
 * - other class template which derives from plask::CacheRemoveStrategyBase and have void onEvent(typename Key::Event& evt) method - custom.
 */
template <typename Key, typename Value, template<typename SKey, typename SValuePtr> class DeleteStrategy = CacheRemoveOnlyWhenDeleted >
struct StrongCache: public CacheBase<Key, plask::shared_ptr<Value>, DeleteStrategy> {

    /**
     * Try get element from cache.
     * @param index key of element
     * @return non-null value from cache stored for key or nullptr if there is no value for given index
     */
    plask::shared_ptr<Value> get(Key* index) const {
        auto iter = this->map.find(index);
        return iter != this->map.end() ? iter->second : plask::shared_ptr<Value>();
    }

    /**
     * Try get element from cache.
     * @param index key of element
     * @return non-null value from cache stored for key or nullptr if there is no value for given index
     */
    plask::shared_ptr<Value> get(plask::shared_ptr<Key> index) const {
        return get(index.get());
    }

};

}   // namespace plask


#endif // CACHE_H
