#ifndef PLASK__MEMORY_H
#define PLASK__MEMORY_H

/** @file
This file includes utils connected with memory managment.
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
 * Typically, subclasses adds some delegates methods to holded object.
 * Holded object is store by pointer, so it can store class dirived from T.
 * Stored object is deleted in destructor.
 * Holder has all asign and copy constructors which use stored object clone() method.
 * @tparam T type of class to hold, must have clone() method which make a copy of object
 */
template <typename T>
struct Holder {
    
    protected:
    
    /// Holded object. Typically can be nullptr only after move assigment.
    T* holded;
    
    public:
    
    /**
     * @brief Construct a holder with given @p holded object.
     * @param holded object to hold, should be not nullptr
     */
    Holder(T* holded): holded(holded) {}
    
    /**
     * @brief Construct a holder with given @p holded object.
     * @param holded object to hold, should be not nullptr
     */
    Holder(T& holded): holded(&holded) {}
    
    /**
     * @brief Copy constructor. Use holded.clone().
     * @param to_copy object to copy
     */
    Holder(const Holder<T>& to_copy): holded(to_copy.holded.clone()) {}
    
    /**
     * @brief Move constructor.
     *
     * It doesn't call holded.clone().
     * @param to_move object to move
     */
    Holder(Holder<T>&& to_move): holded(to_move.holded) { to_move.holded = nullptr; }
    
    /**
     * @brief Copy operator. Use holded.clone().
     * @param to_copy object to copy
     */
    Holder<T>& operator=(const Holder& to_copy) {
        if (holded == to_copy.holded) return;   //self-assigment protection
        delete holded;
        holded = to_copy.holded.clone();
        return *this;
    }
    
    /**
     * @brief Move operator.
     *
     * It doesn't call holded.clone().
     * @param to_move object to move
     */
    Holder<T>& operator=(Holder&& to_move) {
        std::swap(holded, to_move.holded);  //to_move destructor will delete this old holded for a moment
        return *this;
    }
    
    /// Delete holded object using delete.
    ~Holder() { delete holded; }
    
};
}   // namespace plask

#endif // PLASK__MEMORY_H
