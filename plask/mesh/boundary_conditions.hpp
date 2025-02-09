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
#ifndef PLASK__BOUNDARY_CONDITIONS_H
#define PLASK__BOUNDARY_CONDITIONS_H

#include <list>
#include "../exceptions.hpp"
#include "boundary.hpp"

namespace plask {

/// One boundary-condition pair.
template <typename BoundaryT, typename ValueT>
struct BoundaryCondition {
    typedef BoundaryT Boundary;   ///< Boundary type for mesh of type MeshType
    typedef typename Boundary::MeshType MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describe boundary condition

    Boundary place;     ///< Boundary
    ValueType value;    ///< Condition

    /**
     * Construct boundary-condition pair.
     * @param place boundary
     * @param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryCondition(const Boundary& place, ConditionArgumentsTypes&&... value_args)
        : place(place),
          value(std::forward<ConditionArgumentsTypes>(value_args)...) {}

    /**
     * Construct boundary-condition pair.
     * @param place boundary
     * @param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryCondition(Boundary&& place, ConditionArgumentsTypes&&... value_args)
        : place(std::forward<Boundary>(place)),
          value(std::forward<ConditionArgumentsTypes>(value_args)...) {}
};


/// One boundary-condition pair concretized for a given mesh.
template <typename BoundaryT, typename ValueT>
struct BoundaryConditionWithMesh {
    typedef BoundaryT Boundary;   ///< Boundary type for mesh of type MeshType
    typedef typename Boundary::MeshType MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describe boundary condition

    BoundaryNodeSet place;     ///< Set of mesh indexes.
    ValueType value;           ///< Condition value.

    /**
     * Construct boundary-condition pair.
     * @param place boundary
     * @param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryConditionWithMesh(const BoundaryNodeSet& place, ConditionArgumentsTypes&&... value_args)
        : place(place),
          value(std::forward<ConditionArgumentsTypes>(value_args)...) {}

    /**
     * Construct boundary-condition pair.
     * @param place boundary
     * @param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryConditionWithMesh(BoundaryNodeSet&& place, ConditionArgumentsTypes&&... value_args)
        : place(std::move(place)),
          value(std::forward<ConditionArgumentsTypes>(value_args)...) {}
};

template <typename BoundaryT, typename ValueT> struct BoundaryConditions;

/**
 * Set of boundary conditions instances for given mesh type and boundary condition description type.
 * @tparam BoundaryT type of boundary
 * @tparam ValueT type which describe boundary condition
 * @ref boundaries
 */
template <typename BoundaryT, typename ValueT>
struct BoundaryConditionsWithMesh
{
    typedef BoundaryT Boundary;
    typedef typename Boundary::MeshType MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describes boundary condition

    /// One boundary-condition pair.
    typedef BoundaryConditionWithMesh<Boundary, ValueType> Element;

private:
    typedef std::vector<Element> elements_container_t;
    elements_container_t container;
    friend struct BoundaryConditions<BoundaryT,ValueType>;

public:
    typedef typename elements_container_t::iterator iterator;
    typedef typename elements_container_t::const_iterator const_iterator;

    iterator begin() { return container.begin(); }
    const_iterator begin() const { return container.begin(); }

    iterator end() { return container.end(); }
    const_iterator end() const { return container.end(); }

    /**
     * Delegate all constructors to underline container (which is std::list\<Element>).
     * @param args arguments to delegate
     */
    template <typename... ArgsTypes>
    BoundaryConditionsWithMesh(ArgsTypes&&... args): container(std::forward<ArgsTypes>(args)...) {}

    /**
     * Get reference to boundary condition with given @p index.
     *
     * This method has linear time complexity.
     * @param index index of element
     * @return reference to boundary condition with given @p index
     * @throw OutOfBoundsException if @p index is not valid
     */
    Element& operator[](std::size_t index) {
        return container[index];
    }

    /**
     * Get const reference to boundary condition with given @p index.
     *
     * This method has linear time complexity.
     * @param index index of element
     * @return const reference to boundary condition with given @p index
     * @throw OutOfBoundsException if @p index is not valid
     */
    const Element& operator[](std::size_t index) const {
        return container[index];
    }

    /**
     * Get number of elements.
     *
     * This method has linear time complexity.
     * @return number of elements
     */
    std::size_t size() const {
        return container.size();
    }

    /**
     * Check if this is empty.
     *
     * It has constant time complexity.
     * @return @c true only if this container contains no conditions boundaries
     */
    bool empty() const {
        return container.empty();
    }

    /**
     * Check if any boundary contains a @p mesh_index
     *
     * @param mesh_index index in @p mesh
     * @return element which boundary contains @p mesh_index or @ref end() if there is no such element
     */
    const_iterator find(std::size_t mesh_index) const {
        auto i = begin();
        while (i != end() && !i->place.contains(mesh_index)) ++i;
        return i;
    }

    plask::optional<ValueType> getValue(std::size_t mesh_index) const {
        for (auto i: container)
            if (i.place.contains(mesh_index)) return plask::optional<ValueType>(i.value);
        return plask::optional<ValueType>();
    }
};


/**
 * Set of boundary conditions for given mesh type and boundary condition description type.
 * @tparam BoundaryT type of boundary
 * @tparam ValueT type which describe boundary condition
 * @ref boundaries
 */
template <typename BoundaryT, typename ValueT>
struct BoundaryConditions
{
    typedef BoundaryT Boundary;   ///< Boundary type for a mesh of type MeshType
    typedef typename Boundary::MeshType MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describes boundary condition

    /// One boundary-condition pair.
    typedef BoundaryCondition<Boundary, ValueType> Element;

private:
    typedef std::list<Element> elements_container_t;    // std::list does not invalidate iterators on add/erase
    elements_container_t container;

public:
    typedef typename elements_container_t::iterator iterator;
    typedef typename elements_container_t::const_iterator const_iterator;

    iterator begin() { return container.begin(); }
    const_iterator begin() const { return container.begin(); }

    iterator end() { return container.end(); }
    const_iterator end() const { return container.end(); }

private:
    /// @return iterator to last element
    iterator lastIterator() { return --end(); }

public:

    /**
     * Delegate all constructors to underline container (which is std::list\<Element>).
     * @param args arguments to delegate
     */
    template <typename... ArgsTypes>
    BoundaryConditions(ArgsTypes&&... args): container(std::forward<ArgsTypes>(args)...) {}

    /**
     * Get iterator to element with given @p index.
     *
     * This method has linear time complexity.
     * @param[in] index index of element
     * @return iterator to element with given @p index or @c end() if @p index is not valid
     */
    iterator getIteratorForIndex(std::size_t index) {
        iterator result = begin();
        while (index > 0 && result != end()) { ++result; --index; }
        return result;
    }

    /**
     * Get iterator to element with given @p index.
     *
     * This method has linear time complexity.
     * @param[in] index index of element
     * @return iterator to element with given @p index or @c end() if @p index is not valid
     */
    const_iterator getIteratorForIndex(std::size_t index) const {
        const_iterator result = begin();
        while (index > 0 && result != end()) { ++result; --index; }
        return result;
    }

    /**
     * Get reference to boundary condition with given @p index.
     *
     * This method has linear time complexity.
     * @param index index of element
     * @return reference to boundary condition with given @p index
     * @throw OutOfBoundsException if @p index is not valid
     */
    Element& operator[](std::size_t index) {
        iterator i = getIteratorForIndex(index);
        if (i == end()) OutOfBoundsException("BoundaryConditions[]", "index");
        return *i;
    }

    /**
     * Get const reference to boundary condition with given @p index.
     *
     * This method has linear time complexity.
     * @param index index of element
     * @return const reference to boundary condition with given @p index
     * @throw OutOfBoundsException if @p index is not valid
     */
    const Element& operator[](std::size_t index) const {
        const_iterator i = getIteratorForIndex(index);
        if (i == end()) OutOfBoundsException("BoundaryConditions::at", "index");
        return *i;
    }

    /**
     * Add new boundary condition to this (to end of elements list).
     *
     * It doesn't invalidate any iterators. It has constant time complexity.
     * @param element boundary condition to add
     * @return iterator to added element which allow to change or erase added element in future
     */
    iterator add(Element&& element) {
        container.push_back(std::forward<Element>(element));
        return lastIterator();
    }

    /**
     * Add new boundary condition to this (to end of elements list).
     *
     * It doesn't invalidate any iterators. It has constant time complexity.
     * @param place boundary
     * @param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     * @return iterator to added element which allow to change or erase added element in future
     */
    template <typename... ConditionArgumentsTypes>
    iterator add(Boundary&& place, ConditionArgumentsTypes&&... value_args) {
        container.emplace_back(std::forward<Boundary>(place), std::forward<ConditionArgumentsTypes>(value_args)...);
        return lastIterator();
    }

    /**
     * Insert new boundary condition to this at specified position.
     *
     * It doesn't invalidate any iterators. It has constant time complexity.
     * \param index insert position
     * \param element boundary condition to add
     * \return iterator to inserted element which allow to change or erase added element in future
     */
    iterator insert(std::size_t index, Element&& element) {
        iterator i = getIteratorForIndex(index);
        container.insert(i, std::forward<Element>(element));
        return lastIterator();
    }

    /**
     * Insert new boundary condition to this at specified position.
     *
     * It doesn't invalidate any iterators. It has constant time complexity.
     * \param index insert position
     * \param place boundary
     * \param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     * \return iterator to inserted element which allow to change or erase added element in future
     */
    template <typename... ConditionArgumentsTypes>
    iterator insert(std::size_t index, Boundary&& place, ConditionArgumentsTypes&&... value_args) {
        iterator i = getIteratorForIndex(index);
        container.emplace(i, std::forward<Boundary>(place), std::forward<ConditionArgumentsTypes>(value_args)...);
        return lastIterator();
    }

    /// Delete all boundary conditions from this set.
    void clear() {
        container.clear();
    }

    /**
     * Remove the element point by @p to_erase from list.
     *
     * It doesn't invalidate any iterators other than @p to_erase. It has constant time complexity.
     * @param to_erase iterator which show element to erase
     */
    void erase(iterator to_erase) {
        container.erase(to_erase);
    }

    /**
     * Remove the elements in the range [first, last) from list.
     *
     * It doesn't invalidate any iterators from outside of deleted range. It has linear time complexity dependent on the length of the range.
     * @param first, last range of elements to remove
     */
    void erase(iterator first, iterator last) {
        container.erase(first, last);
    }

    /**
     * Remove the element with given @p index from list.
     *
     * Do nothing if @p index is not valid.
     *
     * It doesn't invalidate any iterators which not point to deleted element.
     * It has linear time complexity.
     * @param index index of element to remove
     */
    void erase(std::size_t index) {
        iterator i = getIteratorForIndex(index);
        if (i == end()) OutOfBoundsException("BoundaryConditions[]", "index");
        erase(i);
    }

    /**
     * Get number of elements.
     *
     * This method has linear time complexity.
     * @return number of elements
     */
    std::size_t size() const {
        return container.size();
    }

    /**
     * Check if this is empty.
     *
     * It has constant time complexity.
     * @return @c true only if this container contains no conditions boundaries
     */
    bool empty() const {
        return container.empty();
    }

    /**
     * Get BoundaryConditionsWithMesh<Boundary,ValueType>
     * @param mesh mesh
     */
    BoundaryConditionsWithMesh<Boundary,ValueType> get(const typename Boundary::MeshType& mesh,
                                                       const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        BoundaryConditionsWithMesh<Boundary,ValueType> impl;
        impl.container.reserve(container.size());
        for (auto& el: container) {
            auto place = el.place(mesh, geometry);
            if (place.empty())
                writelog(LOG_WARNING, "Boundary condition with value {} contains no points for given mesh", str(el.value));
            impl.container.push_back(
                BoundaryConditionWithMesh<Boundary,ValueType>(place, el.value)
            );
        }
        return impl;
    }

    /**
     * Get BoundaryConditionsWithMesh<Boundary,ValueType>
     * @param mesh mesh
     * @param geometry geometry at which the boundary conditions are defines
     */
    BoundaryConditionsWithMesh<Boundary,ValueType> get(const shared_ptr<const typename Boundary::MeshType>& mesh,
                                                       const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        return get(*mesh, geometry);
    }

    /**
     * Get BoundaryConditionsWithMesh<Boundary,ValueType>
     * @param mesh mesh
     * @param geometry geometry at which the boundary conditions are defines
     */
    BoundaryConditionsWithMesh<Boundary,ValueType> operator()(const MeshType& mesh,
                                                              const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        return get(mesh, geometry);
    }

    /**
     * Get BoundaryConditionsWithMesh<Boundary,ValueType>
     * @param mesh mesh
     * @param geometry geometry at which the boundary conditions are defines
     */
    BoundaryConditionsWithMesh<Boundary,ValueType> operator()(const shared_ptr<const MeshType>& mesh,
                                                              const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        return get(*mesh, geometry);
    }

};

}   // namespace plask

#endif // PLASK__BOUNDARY_CONDITIONS_H
