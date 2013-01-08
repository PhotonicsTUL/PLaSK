#ifndef PLASK__BOUNDARY_CONDITIONS_H
#define PLASK__BOUNDARY_CONDITIONS_H

#include <list>
#include "../exceptions.h"
#include "boundary.h"

namespace plask {

/// One boundary-condition pair.
template <typename MeshT, typename ValueT>
struct BoundaryCondition {
    typedef MeshT MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describe boundary condition
    typedef typename MeshType::Boundary Boundary;   ///< Boundary type for mesh of type MeshType

    Boundary place;          ///< Boundary
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


/// One boundary-condition pair with mesh.
template <typename MeshT, typename ValueT>
struct BoundaryConditionWithMesh {
    typedef MeshT MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describe boundary condition
    typedef typename MeshType::Boundary::WithMesh Boundary;   ///< Boundary type for mesh of type MeshType

    Boundary place;          ///< Boundary with mesh
    ValueType value;    ///< Condition

    /**
     * Construct boundary-condition pair.
     * @param place boundary
     * @param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryConditionWithMesh(const Boundary& place, ConditionArgumentsTypes&&... value_args)
        : place(place),
          value(std::forward<ConditionArgumentsTypes>(value_args)...) {}

    /**
     * Construct boundary-condition pair.
     * @param place boundary
     * @param value_args arguments for condition constructor, can be just one argument of ValueType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryConditionWithMesh(Boundary&& place, ConditionArgumentsTypes&&... value_args)
        : place(std::forward<Boundary>(place)),
          value(std::forward<ConditionArgumentsTypes>(value_args)...) {}
};

template <typename MeshT, typename ValueT> struct BoundaryConditions;

/**
 * Set of boundary conditions instances for given mesh type and boundary condition description type.
 * @tparam MeshT type of mesh
 * @tparam ValueT type which describe boundary condition
 * @ref boundaries
 */
template <typename MeshT, typename ValueT>
struct BoundaryConditionsWithMesh
{
    typedef MeshT MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describes boundary condition

    /// One boundary-condition pair.
    typedef BoundaryConditionWithMesh<MeshType, ValueType> Element;

private:
    typedef std::vector<Element> elements_container_t;
    elements_container_t container;
    friend struct BoundaryConditions<MeshType,ValueType>;

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
     * @return @c true only if this container includes no conditions boundaries
     */
    bool empty() const {
        return container.empty();
    }

    /**
     * Check if any boundary includes a @p mesh_index
     *
     * @param mesh_index index in @p mesh
     * @return element which boundary includes @p mesh_index or @ref end() if there is no such element
     */
    const_iterator find(std::size_t mesh_index) const {
        auto i = begin();
        while (i != end() && !i->place.includes(mesh_index)) ++i;
        return i;
    }

    boost::optional<ValueType> getValue(std::size_t mesh_index) const {
        for (auto i: container)
            if (i.place.includes(mesh_index)) return boost::optional<ValueType>(i.value);
        return boost::optional<ValueType>();
    }
};


/**
 * Set of boundary conditions for given mesh type and boundary condition description type.
 * @tparam MeshT type of mesh
 * @tparam ValueT type which describe boundary condition
 * @ref boundaries
 */
template <typename MeshT, typename ValueT>
struct BoundaryConditions
{
    typedef MeshT MeshType;    ///< type of mesh
    typedef ValueT ValueType;  ///< type which describes boundary condition
    typedef typename MeshType::Boundary Boundary;   ///< Boundary type for a mesh of type MeshType

    /// One boundary-condition pair.
    typedef BoundaryCondition<MeshType, ValueType> Element;

private:
    typedef std::list<Element> elements_container_t;    // std::list to not invalidate iterators on add/erase
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
     * @return @c true only if this container includes no conditions boundaries
     */
    bool empty() const {
        return container.empty();
    }

    /**
     * Get BoundaryConditionsWithMesh<MeshType,ValueType>
     * @param mesh mesh
     */
    BoundaryConditionsWithMesh<MeshType,ValueType> get(const MeshType& mesh) const {
        BoundaryConditionsWithMesh<MeshType,ValueType> impl;
        impl.container.reserve(container.size());
        for (auto& el: container) impl.container.push_back(BoundaryConditionWithMesh<MeshType,ValueType>(el.place(mesh), el.value));
        return impl;
    }

    /**
     * Get BoundaryConditionsWithMesh<MeshType,ValueType>
     * @param mesh mesh
     */
    BoundaryConditionsWithMesh<MeshType,ValueType> get(const shared_ptr<const MeshType>& mesh) const {
        return get(*mesh);
    }

    /**
     * Get BoundaryConditionsWithMesh<MeshType,ValueType>
     * @param mesh mesh
     */
    BoundaryConditionsWithMesh<MeshType,ValueType> operator()(const MeshType& mesh) const {
        return get(mesh);
    }

    /**
     * Get BoundaryConditionsWithMesh<MeshType,ValueType>
     * @param mesh mesh
     */
    BoundaryConditionsWithMesh<MeshType,ValueType> operator()(const shared_ptr<const MeshType>& mesh) const {
        return get(*mesh);
    }

};

}   // namespace plask

#endif // PLASK__BOUNDARY_CONDITIONS_H
