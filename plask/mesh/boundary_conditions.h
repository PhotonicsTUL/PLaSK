#ifndef PLASK__BOUNDARY_CONDITIONS_H
#define PLASK__BOUNDARY_CONDITIONS_H

#include <list>
#include "../exceptions.h"
#include "boundary.h"

namespace plask {

/// One boundary-condition pair.
template <typename MeshT, typename ConditionT>
struct BoundaryCondition {
    typedef MeshT MeshType;    ///< type of mesh
    typedef ConditionT ConditionType;  ///< type which describe boundary condition
    typedef typename MeshType::Boundary Boundary;   ///< Boundary type for mesh of type MeshType

    Boundary boundary;          ///< Boundary
    ConditionType condition;    ///< Condition

    /**
     * Construct boundary-condition pair.
     * @param boundary boundary
     * @param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryCondition(const Boundary& boundary, ConditionArgumentsTypes&&... conditionsArg)
        : boundary(boundary),
          condition(std::forward<ConditionArgumentsTypes>(conditionsArg)...) {}

    /**
     * Construct boundary-condition pair.
     * @param boundary boundary
     * @param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryCondition(Boundary&& boundary, ConditionArgumentsTypes&&... conditionsArg)
        : boundary(std::forward<Boundary>(boundary)),
          condition(std::forward<ConditionArgumentsTypes>(conditionsArg)...) {}
};

/**
 * Set of boundary conditions for given mesh type and boundary condition description type.
 * @tparam MeshT type of mesh
 * @tparam ConditionT type which describe boundary condition
 * @ref boundaries
 */
template <typename MeshT, typename ConditionT>
struct BoundaryConditions
{
    typedef MeshT MeshType;    ///< type of mesh
    typedef ConditionT ConditionType;  ///< type which describes boundary condition
    typedef typename MeshType::Boundary Boundary;   ///< Boundary type for a mesh of type MeshType

    /// One boundary-condition pair.
    typedef BoundaryCondition<MeshType, ConditionType> Element;

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
     * @throw OutOfBoundException if @p index is not valid
     */
    Element& operator[](std::size_t index) {
        iterator i = getIteratorForIndex(index);
        if (i == end()) OutOfBoundException("BoundaryConditions[]", "index");
        return *i;
    }

    /**
     * Get const reference to boundary condition with given @p index.
     *
     * This method has linear time complexity.
     * @param index index of element
     * @return const reference to boundary condition with given @p index
     * @throw OutOfBoundException if @p index is not valid
     */
    const Element& operator[](std::size_t index) const {
        const_iterator i = getIteratorForIndex(index);
        if (i == end()) OutOfBoundException("BoundaryConditions::at", "index");
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
     * @param boundary boundary
     * @param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
     * @return iterator to added element which allow to change or erase added element in future
     */
    template <typename... ConditionArgumentsTypes>
    iterator add(Boundary&& boundary, ConditionArgumentsTypes&&... conditionsArg) {
        container.emplace_back(std::forward<Boundary>(boundary), std::forward<ConditionArgumentsTypes>(conditionsArg)...);
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
     * \param boundary boundary
     * \param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
     * \return iterator to inserted element which allow to change or erase added element in future
     */
    template <typename... ConditionArgumentsTypes>
    iterator insert(std::size_t index, Boundary&& boundary, ConditionArgumentsTypes&&... conditionsArg) {
        iterator i = getIteratorForIndex(index);
        container.emplace(i, std::forward<Boundary>(boundary), std::forward<ConditionArgumentsTypes>(conditionsArg)...);
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
        if (i == end()) OutOfBoundException("BoundaryConditions[]", "index");
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
     * Check if any boundary includes a @p mesh_index for given @p mesh.
     *
     * Note: if you whish to call includes for more than one point using the same mesh, it is much more effective to call get first
     *      and next call includes on returned object.
     * @param mesh mesh
     * @param mesh_index index in @p mesh
     * @return element which boundary includes @p mesh_index for given @p mesh or @ref end() if there is no such element
     */
    iterator includes(const MeshType& mesh, std::size_t mesh_index) {
        auto i = begin();
        while (i != end() && !i->boundary.includes(mesh, mesh_index)) ++i;
        return i;
    }

    /**
     * Get Boundary<MeshT>::WithMesh for sum of boundaries in this.
     * @param mesh mesh
     */
    typename Boundary::WithMesh get(const MeshType& mesh) const {
        SumBoundaryImpl<MeshType>* impl = new SumBoundaryImpl<MeshType>;
        for (auto& b: container) impl->push_back(b(mesh));
        return typename Boundary::WithMesh(impl);
    }

    /**
     * Get Boundary<MeshT>::WithMesh for sum of boundaries in this.
     * @param mesh mesh
     */
    typename Boundary::WithMesh operator()(const MeshType& mesh) const {
        return get(mesh);
    }
};

}   // namespace plask

#endif // PLASK__BOUNDARY_CONDITIONS_H
