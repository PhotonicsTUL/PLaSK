#ifndef PLASK__MEMORY_H
#define PLASK__MEMORY_H

/** @file
This file contains utils connected with memory managment, it contains utils to operate on memory, pointers, smart pointers, etc.
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
 * Template for base class of object holder. It allows to hold polymorphic class inside one type.
 *
 * Typically, subclasses adds some delegates methods to held object.
 * Held object is stored by pointer, so it can store class derived from T.
 * Stored object is deleted in destructor.
 * Holder has all assign and copy constructors which use stored object clone() method.
 * @tparam T type of class to hold, must have clone() method which make a copy of object
 */
template <typename T>
struct Holder {

    protected:

    /// Held object. Typically can be nullptr only after move assigment.
    T* held;

    public:

    /**
     * @brief Construct a holder with given @p held object.
     * @param held object to hold, should be not nullptr
     */
    Holder(T* held) noexcept: held(held) {}

    /**
     * @brief Construct a holder with given @p held object.
     * @param held object to hold, should be not nullptr
     */
    Holder(T& held) noexcept: held(&held) {}

    /**
     * @brief Copy constructor. Use held->clone().
     * @param to_copy object to copy
     */
    Holder(const Holder<T>& to_copy): held(to_copy.held->clone()) {}

    /**
     * @brief Move constructor.
     *
     * It doesn't call held.clone().
     * @param to_move object to move
     */
    Holder(Holder<T>&& to_move) noexcept: held(to_move.held) { to_move.held = nullptr; }

    /**
     * @brief Copy operator. Use held->clone().
     * @param to_copy object to copy
     */
    Holder<T>& operator=(const Holder& to_copy) {
        if (held == to_copy.held) return *this;   //self-assigment protection
        delete held;
        held = to_copy.held->clone();
        return *this;
    }

    /**
     * @brief Move operator.
     *
     * It doesn't call held.clone().
     * @param to_move object to move
     */
    Holder<T>& operator=(Holder&& to_move) noexcept {
        std::swap(held, to_move.held);  // to_move destructor will delete this old held for a moment
        return *this;
    }

    /// Delete held object using delete.
    ~Holder() { delete held; }

};

/**
 * Template for base class of object holder. It allow to hold polymorphic class inside one type and use reference counting to delete held object in proper time.
 *
 * Typically, subclasses adds some delegates methods to held object.
 * Hold object is stored by pointer, so it can store class derived from T.
 * Stored object is deleted in destructor when last reference to it is lost.
 * @tparam T type of class to hold
 */
template <typename T>
struct HolderRef {

    protected:

    /// Hold object. Typically can be nullptr only after move assigment.
    shared_ptr<T> held;

    public:

    HolderRef() {}

    /**
     * @brief Construct a holder with given @p held object.
     * @param held object to hold
     */
    HolderRef(T* held): held(held) {}

    bool isNotNull() const { return held != nullptr; }

    bool isNull() const { return !held; }

    void reset(T* new_held) { this->held.reset(new_held); }

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
