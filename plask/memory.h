#ifndef PLASK__MEMORY_H
#define PLASK__MEMORY_H

/** @file
This file includes utils connected with memory managment, it includes utils to operate on memory, pointers, smart pointers, etc.
It put smart pointers (boost or std ones - dependly of plask build configuration) in plask namespace.
*/

#include <plask/config.h>

#ifdef PLASK_SHARED_PTR_STD

#include <memory>
namespace plask {
    using std::shared_ptr;
    using std::make_shared;
    using std::dynamic_pointer_cast;
    using std::static_pointer_cast;
    using std::const_pointer_cast;
    using std::weak_ptr;
    using std::enable_shared_from_this;
}

#else // PLASK_SHARED_PTR_STD
// Use boost::shared_ptr

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
namespace plask {
    using boost::shared_ptr;
    using boost::make_shared;
    using boost::dynamic_pointer_cast;
    using boost::static_pointer_cast;
    using boost::const_pointer_cast;
    using boost::weak_ptr;
    using boost::enable_shared_from_this;
}

#endif // PLASK_SHARED_PTR_STD


namespace plask {
/**
 * Template for base class of object holder. It allow to hold polimorhic class inside one type.
 *
 * Typically, subclasses adds some delegates methods to hold object.
 * Holded object is store by pointer, so it can store class dirived from T.
 * Stored object is deleted in destructor.
 * Holder has all asign and copy constructors which use stored object clone() method.
 * @tparam T type of class to hold, must have clone() method which make a copy of object
 */
template <typename T>
struct Holder {

    protected:

    /// Holded object. Typically can be nullptr only after move assigment.
    T* hold;

    public:

    /**
     * @brief Construct a holder with given @p hold object.
     * @param hold object to hold, should be not nullptr
     */
    Holder(T* hold): hold(hold) {}

    /**
     * @brief Construct a holder with given @p hold object.
     * @param hold object to hold, should be not nullptr
     */
    Holder(T& hold): hold(&hold) {}

    /**
     * @brief Copy constructor. Use hold.clone().
     * @param to_copy object to copy
     */
    Holder(const Holder<T>& to_copy): hold(to_copy.hold.clone()) {}

    /**
     * @brief Move constructor.
     *
     * It doesn't call hold.clone().
     * @param to_move object to move
     */
    Holder(Holder<T>&& to_move): hold(to_move.hold) { to_move.hold = nullptr; }

    /**
     * @brief Copy operator. Use hold.clone().
     * @param to_copy object to copy
     */
    Holder<T>& operator=(const Holder& to_copy) {
        if (hold == to_copy.hold) return;   //self-assigment protection
        delete hold;
        hold = to_copy.hold.clone();
        return *this;
    }

    /**
     * @brief Move operator.
     *
     * It doesn't call hold.clone().
     * @param to_move object to move
     */
    Holder<T>& operator=(Holder&& to_move) {
        std::swap(hold, to_move.hold);  //to_move destructor will delete this old hold for a moment
        return *this;
    }

    /// Delete hold object using delete.
    ~Holder() { delete hold; }

};

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

}   // namespace plask

#endif // PLASK__MEMORY_H
