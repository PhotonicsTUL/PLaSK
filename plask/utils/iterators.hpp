/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__UTILS_ITERATORS_H
#define PLASK__UTILS_ITERATORS_H

/** @file
This file contains iterators utils.
*/

//general iterators utils

#include <boost/iterator/iterator_facade.hpp>
#include <type_traits>
#include <memory>

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

    // some typedefs compatible with stl:

    /// Type of objects pointed by the iterator.
    typedef ValueT value_type;

    /// Type to represent a reference to an object pointed by the iterator.
    typedef ReferenceT reference;

    ///@return current value
    virtual ReferenceT dereference() const = 0;

    /// Iterate to next value.
    virtual void increment() = 0;

    /// Virtual destructor, do nothing.
    virtual ~PolymorphicForwardIteratorImpl() {}

    /**
     * Check if this is equal to @p other.
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
     * @return clone of @c *this, allocated with @a new operator (caller is responsible for deleting it)
     */
    virtual std::unique_ptr<PolymorphicForwardIteratorImpl<ValueT, ReferenceT>> clone() const = 0;
};

/**
Base class for forward, polymorphic iterators implementations which allow to get index of current position.

It is similar to PolymorphicForwardIteratorImpl but has getIndex method.
@tparam ValueT Type to iterate over.
@tparam ReferenceT Type returned by dereference operation.
Note that default type is not good if dereference returns temporary object.
In such case <code>const ValueT</code> can be a better choice.
*/
template <typename ValueT, typename ReferenceT = ValueT&>
struct PolymorphicForwardIteratorWithIndexImpl: public PolymorphicForwardIteratorImpl<ValueT, ReferenceT> {

    /**
     * Get index for current iterator state.
     * @return index for current iterator state
     */
    virtual std::size_t getIndex() const = 0;
};

/**
 * Polymorphic iterator that wrap another (usually not polymorphic) iterator (of type wrapped_iterator_type).
 */
template <typename wrapped_iterator_type, typename ValueT, typename ReferenceT = ValueT&>
struct PolymorphicForwardIteratorWrapperImpl: public PolymorphicForwardIteratorImpl<ValueT, ReferenceT> {

    wrapped_iterator_type wrapped_iterator;

    PolymorphicForwardIteratorWrapperImpl(wrapped_iterator_type wrapped_iterator): wrapped_iterator(std::move(wrapped_iterator)) {}

    ReferenceT dereference() const override { return *wrapped_iterator; }

    void increment() override { ++wrapped_iterator; }

    bool equal(const PolymorphicForwardIteratorImpl<ValueT, ReferenceT>& other) const override {
        return wrapped_iterator == static_cast<const PolymorphicForwardIteratorWrapperImpl<wrapped_iterator_type, ValueT, ReferenceT>&>(other).wrapped_iterator;
    }

    std::unique_ptr<PolymorphicForwardIteratorImpl<ValueT, ReferenceT>> clone() const override {
        return std::unique_ptr<PolymorphicForwardIteratorImpl<ValueT, ReferenceT>>(
                new PolymorphicForwardIteratorWrapperImpl<wrapped_iterator_type, ValueT, ReferenceT>(wrapped_iterator));
    }

};

/**
Polymorphic, forward iterator.

Hold and delegate all calls to implementation object which is a instantiation of PolymorficForwardIteratorImpl template.

@tparam ImplT instantiation of PolymorphicForwardIteratorImpl
*/
template <typename ImplT>
struct PolymorphicForwardIterator:
    public boost::iterator_facade<
        PolymorphicForwardIterator<ImplT>,
        typename ImplT::value_type,
        boost::forward_traversal_tag,
        typename ImplT::reference
    > {

    protected:
    std::unique_ptr<ImplT> impl;

    public:

    /**
     * Construct iterator which holds given implementation object.
     * @param impl Implementation object. It will be delete by constructor of this.
     *             If it is @c nullptr you should not call any methods of this before assign.
     */
    PolymorphicForwardIterator(ImplT* impl = nullptr): impl(impl) {}

    /**
     * Copy constructor. Clone implementation object.
     * @param src Iterator from which implementation object should be clone.
     */
    PolymorphicForwardIterator(const PolymorphicForwardIterator& src) { impl = src.impl ? src.impl->clone() : nullptr; }

    /**
     * Move constructor.
     * @param src Iterator to move from.
     */
    PolymorphicForwardIterator(PolymorphicForwardIterator&& src) = default;

    /**
     * Swap values of @c this and @p to_swap.
     * @param to_swap
     */
    void swap(PolymorphicForwardIterator & to_swap) noexcept { std::swap(this->impl, to_swap.impl); }

    PolymorphicForwardIterator& operator=(const PolymorphicForwardIterator &src) {
        this->impl = src.impl->clone();
        return *this;
    }

    PolymorphicForwardIterator& operator=(PolymorphicForwardIterator &&src) = default;


    private: //--- methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;
    template <class> friend struct PolymorphicForwardIterator;

    bool equal(const PolymorphicForwardIterator<ImplT>& other) const {
        return impl->equal(*other.impl);
    }

    void increment() {
        impl->increment();
    }

    typename ImplT::reference dereference() const { return impl->dereference(); }

    //TODO use advance?
};

/**
Polymorphic, forward iterator which allow to get index of current position.

Hold and delegate all calls to implementation object which is a instantiation of PolymorphicForwardIteratorWithIndexImpl template.

@tparam ImplT instantiation of PolymorphicForwardIteratorWithIndexImpl
*/
template <typename ImplT>
struct PolymorphicForwardIteratorWithIndex: public PolymorphicForwardIterator<ImplT> {

    /**
     * Construct iterator which holds given implementation object.
     * @param impl Implementation object. It will be delete by constructor of this.
     *             If it is @c nullptr you should not call any methods of this before assign.
     */
    PolymorphicForwardIteratorWithIndex(ImplT* impl = nullptr): PolymorphicForwardIterator<ImplT>(impl) {}

    /**
     * Get index for current iterator state.
     * @return index for current iterator state
     */
    std::size_t getIndex() const {
        return this->impl->getIndex();
    }
};


template <typename ContainerType, typename Reference>
inline Reference dereferenceByIndexOperator(ContainerType& container, std::size_t index) { return container[index]; }

/**
 * Template to create iterators for containers which have operator[].
 * @tparam ContainerType type of container (can be const or non-const)
 * @tparam Reference iterator reference type, should be the same type which return container operator[]
 * In most cases it can be auto-deduced from ContainerType, but only if ContainerType is fully defined and known.
 * @tparam dereference_f function which perform dereference on container, by default calls operator[]
 *
 * Example:
 * @code
 * struct MyIntContainer {
 *   //Typdefes for iterator types:
 *   //(note that second template parameter can't be auto-deduced because MyIntContainer is not fully defined yet)
 *   typedef IndexedIterator<MyRandomAccessContainer, int&> iterator;
 *   typedef IndexedIterator<const MyRandomAccessContainer, const int&> const_iterator;
 *
 *   //begin() and end() methods (non-const and const versions):
 *   iterator begin() { return iterator(0); } //0 is begin index
 *   const_iterator begin() const { return const_iterator(0); } //0 is begin index
 *   iterator end() { return iterator(size()); } //size() is end index
 *   const_iterator end() const { return const_iterator(size()); } //size() is end index
 *
 *   //Methods to get object by index, etc.:
 *   int& operator[](std::size_t index) {  //used by iterator
 *      //code which returns object with given index
 *   }
 *   const int& operator[](std::size_t index) const {  //used by const_iterator
 *      //code which returns object with given index
 *   }
 *   std::size_t size() const { //used by end()
 *      //code which returns number of objects
 *   }
 * };
 * @endcode
 */
template <
    typename ContainerType,
    typename Reference = decltype(std::declval<ContainerType>()[0]),
    Reference (*dereference_f)(ContainerType& container, std::size_t index) = dereferenceByIndexOperator<ContainerType, Reference>>
struct IndexedIterator: public boost::iterator_facade< IndexedIterator<ContainerType, Reference, dereference_f>, typename std::remove_reference<Reference>::type, boost::random_access_traversal_tag, Reference > {

    /// Pointer to container over which we iterate.
    ContainerType* container;

    /// Current iterator position (index).
    std::size_t index;

    /// Construct uninitialized iterator. Don't use it before initialization.
    IndexedIterator() {}

    /**
     * Construct iterator which point to given @a index in given @a container.
     * @param container container to iterate over
     * @param index index in @a container
     */
    IndexedIterator(ContainerType* container, std::size_t index): container(container), index(index) {}

    /**
     * Get current iterator position (index).
     * @return current iterator position (index)
     */
    std::size_t getIndex() const { return index; }

    private: //--- methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;

    template <typename OtherT>
    bool equal(const OtherT& other) const {
        return index == other.index;
    }

    void increment() { ++index; }

    void decrement() { --index; }

    void advance(std::ptrdiff_t to_add) { index = size_t(index + to_add); }

    template <typename OtherT>
    std::ptrdiff_t distance_to(OtherT z) const { return std::ptrdiff_t(z.index) - std::ptrdiff_t(index); }

    Reference dereference() const { return dereference_f(*container, index); }

};

/**
 * Get IndexedIterator for given container.
 * @param c container
 * @param index initial iterator position
 * @return iterator over container @a c which points to the position @a index
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
    typename Reference = decltype(std::declval<FunctorType>()(0)),
    typename Value = typename std::remove_reference<Reference>::type>
struct FunctorIndexedIterator: public boost::iterator_facade< FunctorIndexedIterator<FunctorType, Reference, Value>, Value, boost::random_access_traversal_tag, Reference > {

    /// Functor
    FunctorType functor;

    /// Current iterator position (index).
    std::size_t index;

    /**
     * Construct iterator which point to given in index in given container.
     * @param functor functor
     * @param index index for which this iterator should refer, argument for functor
     */
    FunctorIndexedIterator(FunctorType functor, std::size_t index): functor(functor), index(index) {}

    /**
     * Get current iterator position (index).
     * @return current iterator position (index)
     */
    std::size_t getIndex() const { return index; }

    private: //--- methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;
    template <class, class, class> friend struct FunctorIndexedIterator;

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
 * Template to create iterators which using method having index argument.
 * @tparam ContainerType type of container (can be const or non-const)
 * @tparam ReturnedType exact type returned by container method Method
 * @tparam Method method used to get value
 * @tparam Reference iterator reference type, should be the same type which return functor operator()
 * @tparam Value iterator value type, should be the same type which return container operator[] but without reference
 */
template <
    typename ContainerType,
    typename ReturnedType,
    ReturnedType (ContainerType::*Method)(std::size_t),
    typename Reference = ReturnedType,
    typename Value = typename std::remove_reference<Reference>::type>
struct MethodIterator: public boost::iterator_facade< MethodIterator<ContainerType, ReturnedType, Method, Value, Reference>, Value, boost::random_access_traversal_tag, Reference > {

    /// Pointer to container over which we iterate.
    ContainerType* container;

    /// Current iterator position (index).
    std::size_t index;

    /// Construct uninitialized iterator. Don't use it before initialization.
    MethodIterator() {}

    /**
     * Construct iterator which point to given @a index in given @a container.
     * @param container container to iterate over
     * @param index index in @a container
     */
    MethodIterator(ContainerType* container, std::size_t index): container(container), index(index) {}

    /**
     * Get current iterator position (index).
     * @return current iterator position (index)
     */
    std::size_t getIndex() const { return index; }

    private: //--- methods used by boost::iterator_facade: ---
    friend class boost::iterator_core_access;
	template <
        typename _ContainerType,
        typename _ReturnedType,
        _ReturnedType(_ContainerType::*_Method)(std::size_t),
        typename _Reference,
        typename _Value>
	friend struct MethodIterator;

    template <typename OtherT>
    bool equal(const OtherT& other) const {
        return index == other.index;
    }

    void increment() { ++index; }

    void decrement() { --index; }

    void advance(std::ptrdiff_t to_add) { index += to_add; }

    template <typename OtherT>
    std::ptrdiff_t distance_to(OtherT z) const { return z.index - index; }

    Reference dereference() const { return (container->*Method)(index); }

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

/**
 * ReindexedContainer instantiation is class which objects have reference to original container and operator[].
 * All calls to operator[] are delegated to original container, but argument of call is changed (reindexed) using formula: firstIndex + given_index * delta
 * where:
 * - given_index is call parameter,
 * - firstIndex and delta are paremeters stored in ReindexedContainer.
 *
 * @tparam ContainerType type of original container
 */
template <typename ContainerType>
struct ReindexedContainer {

    ContainerType& originalContainer;

    int firstIndex, delta;

    ReindexedContainer(ContainerType& originalContainer, int firstIndex = 0, int delta = 1)
    : originalContainer(originalContainer), firstIndex(firstIndex), delta(delta) {}

    auto operator[](const std::size_t& this_index) -> decltype(originalContainer[0]) {
        return originalContainer[firstIndex + this_index * delta];
    }

    auto operator[](const std::size_t& this_index) const -> decltype(const_cast<const ContainerType&>(originalContainer)[0])  {
        return originalContainer[firstIndex + this_index * delta];
    }

};

/**
 * Helper function to create ReindexedContainer instantiation objects.
 * @param originalContainer, firstIndex, delta ReindexedContainer constructor parameters
 * @return ReindexedContainer<ContainerType>(originalContainer, firstIndex, delta)
 * @tparam ContainerType type of original container
 */
template <typename ContainerType>
ReindexedContainer<ContainerType> reindexContainer(ContainerType& originalContainer, int firstIndex = 0, int delta = 1) {
    return ReindexedContainer<ContainerType>(originalContainer, firstIndex, delta);
}

}       // namespace plask

#endif
