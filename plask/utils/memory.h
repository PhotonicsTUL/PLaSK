#ifndef PLASK__MEMORY_H
#define PLASK__MEMORY_H

/** @file
This file includes utils to operate on memory, pointers, smart pointers, etc.
*/

#include <config.h>

namespace plask {

/**
 * Copy @a ptr data if is not the only shared_ptr instance managing the current object, i.e. whether ptr.unique() is @c false (ptr.use_count() != 1).
 * @param ptr shared pointer
 * @return @a ptr if ptr.unique() is @c true (ptr.use_count() == 1) or shared_ptr with copy of object managing by @a ptr if ptr.unique() is @c false
 * @tparam T type of object managing by @a ptr, must have copy constructor
 */
template <typename T>
inline shared_ptr<T> getUnique(const shared_ptr<T>& ptr) {
    return ptr.unique() ? ptr : new T(*ptr);
}

/**
 * Copy @a ptr data if is not the only shared_ptr instance managing the current object, i.e. whether ptr.unique() is @c false (ptr.use_count() != 1).
 * @param ptr shared pointer
 * @return @a ptr if ptr.unique() is @c true (ptr.use_count() == 1) or shared_ptr with copy of object managing by @a ptr if ptr.unique() is @c false
 * @tparam T type of object managing by @a ptr, must have copy constructor
 */
template <typename T>
inline shared_ptr<T> getUnique(const shared_ptr<const T>& ptr) {
    return ptr.unique() ? ptr : new T(*ptr);
}

/**
 * Copy @a ptr data if is not the only shared_ptr instance managing the current object, i.e. whether ptr.unique() is @c false (ptr.use_count() != 1).
 * @param ptr shared pointer which will be changed to shared_ptr with copy of object managing by @a ptr if ptr.unique() is @c false
 * @tparam T type of object managing by @a ptr, must have copy constructor
 */
template <typename T>
inline void makeUnique(shared_ptr<T>& ptr) {
    ptr = getUnique(ptr);
}

}       // namespace plask

#endif