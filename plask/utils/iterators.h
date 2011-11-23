#ifndef PLASK__ITERATORS_H
#define PLASK__ITERATORS_H

//general iterators utils

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
    
    /**
     * Move iterator @a distanse steps forward. By default call increment @a distanse times.
     * @param distanse how many steps
     */
    virtual void advance(std::size_t distanse) {
        while (distanse) { increment(); --distanse; }
    }
    
    //a.
        
    //@return @c true only if there are more points to iterate over
    //virtual void hasNext() const = 0;
        
    ///@return clone of @c *this
    virtual PolymorphicForwardIteratorImpl<T>* clone() const = 0;
        
    //Do nothing.
    virtual ~PolymorphicForwardIteratorImpl() {}
};
    
/**
Polymorphic, forward iterator over value with type T.
    
Hold and delgate all calls to PolymorficForwardIteratorImpl<T>.

@tparam T type to iterate over (type returned by dereference operation)
*/
template <typename T>
struct PolymorphicForwardIterator {
        
    PolymorphicForwardIteratorImpl<T>* impl;
        
    public:
        
    PolymorphicForwardIterator(PolymorphicForwardIteratorImpl<T>* impl): impl(impl) {}
    
    ///Delete wrapped iterator.
    ~PolymorphicForwardIterator() { delete impl; }
        
    //Copy constructor
    PolymorphicForwardIterator(const PolymorphicForwardIterator& src) { impl = src.impl->clone(); }
        
    //Move constructor
    PolymorphicForwardIterator(PolymorphicForwardIterator &&src): impl(src.impl) { src.impl = 0; }
        
    T operator*() const { return impl->dereference(); }
    
    bool operator==(const PolymorphicForwardIterator& other) const {
        return impl->equal(other.impl);
    }
    
    bool operator!=(const PolymorphicForwardIterator& other) const {
        return !impl->equal(other.impl);
    }
    
    //pre-increment
    PolymorphicForwardIterator<T>& operator++() {
        impl->increment();
        return *this;
    }
    
    //post-increment
    PolymorphicForwardIterator<T>& operator++(int) {
        PolymorphicForwardIterator<T> result(*this);
        impl->increment();
        return result;
    }
    
    //TODO use advance
    
};


}       //namespace plask
    
#endif
