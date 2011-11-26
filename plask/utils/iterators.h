#ifndef PLASK__ITERATORS_H
#define PLASK__ITERATORS_H

//general iterators utils

#include <boost/iterator/iterator_facade.hpp>

namespace plask {

/**
Base class for forward, polymorphic iterators implementations.
@tparam T type to iterate over (type returned by dereference operation)
*/
template <typename T>
struct PolymorphicForwardIteratorImpl {
    
    ///@return current value
    virtual T dereference() const = 0;
        
    ///Iterate to next value.
    virtual void increment() = 0;
    
    /**
     * @return true only if this is equal to @a other
     */
    virtual bool equal(const PolymorphicForwardIteratorImpl& other) const = 0;
    
    /*
     * Move iterator @a distanse steps forward. By default call increment @a distanse times.
     * @param distanse how many steps
     */
    /*virtual void advance(std::size_t distanse) {
        while (distanse) { increment(); --distanse; }
    }*/
        
    ///@return clone of @c *this
    virtual PolymorphicForwardIteratorImpl<T>* clone() const = 0;
        
    //Do nothing.
    virtual ~PolymorphicForwardIteratorImpl() {}
};
    
/**
Polymorphic, forward iterator over value with type T.
    
Hold and delgate all calls to implementation object which is a PolymorficForwardIteratorImpl<T> type.

@tparam T type to iterate over (type returned by dereference operation)
*/
//TODO operator* return reference
template <typename T>
struct PolymorphicForwardIterator: public boost::iterator_facade< PolymorphicForwardIterator<T>, T, boost::forward_traversal_tag > {
        
    PolymorphicForwardIteratorImpl<T>* impl;
        
    public:
    
    /**
     * Construct iterator which hold given implementation object.
     * @param impl Implementation object. It will be delete by constructor of this.
     *             If it is @c nullptr you should not call any methods of this before assign 
     */
    PolymorphicForwardIterator(PolymorphicForwardIteratorImpl<T>* impl = nullptr): impl(impl) {}
    
    ///Delete wrapped iterator object.
    ~PolymorphicForwardIterator() { delete impl; }
        
    /**
     * Copy constructor. Clone implementation object.
     * @param src Iterator from which implementation object should be clone. It mustn't hold @c nullptr.
     */
    PolymorphicForwardIterator(const PolymorphicForwardIterator& src) { impl = src.impl->clone(); }
        
    /**
     * Move constructor.
     * Move ownership of wrapped implementation object from @a src to this.
     * @param src iterator from which implementation object should be moved
     */
    PolymorphicForwardIterator(PolymorphicForwardIterator &&src): impl(src.impl) { src.impl = 0; }
    
    private: //--- implement methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;
    template <class> friend class PolymorphicForwardIterator;

    bool equal(const PolymorphicForwardIterator<T>& other) const {
        return impl->equal(other.impl);
    }

    void increment() {
        impl->increment();
    }

    T dereference() const { return impl->dereference(); }

    //TODO use advance?
};

/*
 * Template to create iterators for containers which have operator[].
 */
/*template <typename ContainerType, typename value_type = ContainerType::value_type>
struct IndexedIterator: public boost::iterator_facade< IndexedIterator<ContainerType>, value_type, boost::random_access_traversal_tag > {

};*/

}       //namespace plask
    
#endif
