#ifndef PLASK__UTILS_ITERATORS_H
#define PLASK__UTILS_ITERATORS_H

/** @file
This file includes iterators utils.
*/

//general iterators utils

#include <boost/iterator/iterator_facade.hpp>
#include <type_traits>

namespace plask {

/**
Base class for forward, polymorphic iterators implementations.
@tparam ValueT Type to iterate over.
@tparam ReferenceT Type returned by dereference operation.
Note that default type is not good if dereference returns temporary object.
In such case <code>const ValueT</code> can be a better choice.
*/
template <typename ValueT, typename ReferenceT = ValueT&>
struct PolymorphicForwardIteratorImpl {

    //some typedefs compatibile with stl:

    ///Type of elements pointed by the iterator.
    typedef ValueT value_type;

    ///Type to represent a reference to an element pointed by the iterator.
    typedef ReferenceT reference;

    ///@return current value
    virtual ReferenceT dereference() const = 0;

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

    /**
     * Clone this iterator.
     * @return clone of @c *this, reserved by @a new operator (clone caller must trust to delete it)
     */
    virtual PolymorphicForwardIteratorImpl<ValueT, ReferenceT>* clone() const = 0;

    //Do nothing.
    virtual ~PolymorphicForwardIteratorImpl() {}
};

/**
Polymorphic, forward iterator.

Hold and delegate all calls to implementation object which is a specialization of PolymorficForwardIteratorImpl template.

@tparam ImplT specialization of PolymorphicForwardIteratorImpl
*/
template <typename ImplT>
struct PolymorphicForwardIterator:
    public boost::iterator_facade<
        PolymorphicForwardIterator<ImplT>,
        typename ImplT::value_type,
        boost::forward_traversal_tag,
        typename ImplT::reference
    > {

    ImplT* impl;

    public:

    /**
     * Construct iterator which hold given implementation object.
     * @param impl Implementation object. It will be delete by constructor of this.
     *             If it is @c nullptr you should not call any methods of this before assign
     */
    PolymorphicForwardIterator(ImplT* impl = nullptr): impl(impl) {}

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

    private: //--- methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;
    template <class> friend class PolymorphicForwardIterator;

    bool equal(const PolymorphicForwardIterator<ImplT>& other) const {
        return impl->equal(other.impl);
    }

    void increment() {
        impl->increment();
    }

    typename ImplT::reference dereference() const { return impl->dereference(); }

    //TODO use advance?
};

/**
 * Template to create iterators for containers which have operator[].
 * @tparam ContainerType type of container (can be const or non-const)
 * @tparam Reference iterator reference type, should be the same type which return container operator[]
 * @tparam Value iterator value type, should be the same type which return container operator[] but without reference
 */
template <
    typename ContainerType,
    typename Reference = decltype((((ContainerType*)0)->*(&ContainerType::operator[]))(0)),
    typename Value = typename std::remove_reference<Reference>::type>
struct IndexedIterator: public boost::iterator_facade< IndexedIterator<ContainerType, Value, Reference>, Value, boost::random_access_traversal_tag, Reference > {

    ///Pointer to container over which we iterate.
    ContainerType* container;

    ///Current iterator position (index).
    std::size_t index;

    ///Construct uninitialized iterator. Don't use it before initialization.
    IndexedIterator() {}

    /**
     * Construct iterator which point to given @a index in given @a container.
     * @param container container to iterate over
     * @param index index in @a container
     */
    IndexedIterator(ContainerType* container, std::size_t index): container(container), index(index) {}

    private: //--- methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;
    template <class, class, class> friend class IndexedIterator;

    template <typename OtherT>
    bool equal(const OtherT& other) const {
        return index == other.index;
    }

    void increment() { ++index; }

    void decrement() { --index; }

    void advance(std::ptrdiff_t to_add) { index += to_add; }

    template <typename OtherT>
    std::ptrdiff_t distance_to(OtherT z) const { return z.index - index; }

    Reference dereference() const { return (*container)[index]; }

};

/**
 * Get IndexedIterator for given container.
 * @param c container
 * @param index initial iterator position
 * @return iterator over container @a c with position @a index
 * @see @ref IndexedIterator
 */
template <typename ContainerType>
inline IndexedIterator<ContainerType> makeIndexedIterator(ContainerType* c, std::size_t index) {
    return IndexedIterator<ContainerType>(c, index);
}

/**
 * Template to create iterators which using functor having size argument.
 * @tparam ContainerType type of container (can be const or non-const)
 * @tparam Reference iterator reference type, should be the same type which return functor operator()
 * @tparam Value iterator value type, should be the same type which return container operator[] but without reference
 */
template <typename FunctorType,
    typename Reference = decltype((((FunctorType*)0)->*(&FunctorType::operator()))(0)),
    typename Value = typename std::remove_reference<Reference>::type>
struct FunctorIndexedIterator: public boost::iterator_facade< FunctorIndexedIterator<FunctorType, Value, Reference>, Value, boost::random_access_traversal_tag, Reference > {

    ///Functor
    FunctorType functor;

    ///Current iterator position (index).
    std::size_t index;

    /**
     * Construct iterator which point to given in index in given container.
     * @param functor functor
     * @param index index for which this iterator should refere, argument for functor
     */
    FunctorIndexedIterator(FunctorType functor, std::size_t index): functor(functor), index(index) {}

    private: //--- methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;
    template <class, class, class> friend class IndexedIterator;

    template <typename OtherT>
    bool equal(const OtherT& other) const {
        return index == other.index;
    }

    void increment() { ++index; }

    void decrement() { --index; }

    void advance(std::ptrdiff_t to_add) { index += to_add; }

    template <typename OtherT>
    std::ptrdiff_t distance_to(OtherT z) const { return z.index - index; }

    Reference dereference() const { return functor(index); }

};

/**
 * Get FunctorIndexedIterator for given functor.
 * @param f functor
 * @param index initial iterator position
 * @return iterator which using functor @a f and has position @a index
 * @see @ref FunctorIndexedIterator
 */
template <typename Functor>
inline FunctorIndexedIterator<Functor> makeFunctorIndexedIterator(Functor f, std::size_t index) {
    return FunctorIndexedIterator<Functor>(f, index);
}

}       //namespace plask

#endif
