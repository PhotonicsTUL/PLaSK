#ifndef PLASK__BOUNDARY_CONDITIONS_H
#define PLASK__BOUNDARY_CONDITIONS_H

#include "boundary.h"
#include <list>
#include "../exceptions.h"

namespace plask {

/// One boundary-condition pair.
template <typename MeshT, typename ConditionT>
struct BoundaryCondition {
    typedef MeshT MeshType;    ///< type of mesh
    typedef ConditionT ConditionType;  ///< type which describe boundary condition
    typedef typename MeshType::Boundary Boundary;   ///< Boundary type for mesh of type MeshType
    
    Boundary boundary;    ///< Boundary
    ConditionType condition;        ///< Condition

    /**
     * Construct boundary-condition pair.
     * @param boundary boundary
     * @param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    BoundaryCondition(Boundary&& boundary, ConditionArgumentsTypes&&... conditionsArg)
        :boundary(std::forward<boundary>(boundary)),
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
    typedef ConditionT ConditionType;  ///< type which describe boundary condition
    typedef typename MeshType::Boundary Boundary;   ///< Boundary type for mesh of type MeshType

    /// One boundary-condition pair.
    typedef BoundaryCondition<MeshType, ConditionType> Element;

private:
    typedef std::list<Element> elements_container_t;    //std::list to not invalidate iterators on add/erase
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
     * @param index index of element
     * @return reference to boundary condition with given @p index
     * @throw OutOfBoundException if @p index is not valid
     */
    Element& at(std::size_t index) {
        iterator i = getIteratorForIndex(index);
        if (i == end()) OutOfBoundException("BoundaryConditions::at", "index");
        return *i;
    }
    
    /**
     * Get const reference to boundary condition with given @p index.
     * @param index index of element
     * @return const reference to boundary condition with given @p index
     * @throw OutOfBoundException if @p index is not valid
     */
    const Element& at(std::size_t index) const {
        const_iterator i = getIteratorForIndex(index);
        if (i == end()) OutOfBoundException("BoundaryConditions::at", "index");
        return *i;
    }

    /**
     * Add new boundary condidion to this (to end of elements list).
     *
     * It doesn't invalidate any iterators.
     * @param element boundary condidion to add
     * @return iterator to added element which allow to change or erase added element in future
     */
    iterator add(Element&& element) {
        container.push_back(std::forward<Element>(element));
        return lastIterator();
    }

    /**
     * Add new boundary condidion to this (to end of elements list).
     *
     * It doesn't invalidate any iterators.
     * @param boundary boundary
     * @param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
     * @return iterator to added element which allow to change or erase added element in future
     */
    template <typename... ConditionArgumentsTypes>
    iterator add(Boundary&& boundary, ConditionArgumentsTypes&&... conditionsArg) {
        container.emplace_back(std::forward<Boundary>(boundary), std::forward<ConditionArgumentsTypes>(conditionsArg)...);
        return lastIterator();
    }

    /// Delete all boundaries conditions from this set.
    void clear() {
        container.clear();
    }
    
    /**
     * Remove from list the element point by @p to_erase.
     *
     * It doesn't invalidate any iterators other than @p to_erase.
     * @param to_erase iterator which show element to erase
     */
    void erase(const_iterator to_erase) {
        container.erase(to_erase);
    }
    
    /**
     * Remove from list the elements in the range [first, last).
     *
     * It doesn't invalidate any iterators from outside of deleted range.
     * @param first, last range of elements to remove
     */
    void erase(const_iterator first, const_iterator last) {
        container.erase(first, last);
    }
    
    /**
     * Remove from list the element with given @p index.
     *
     * Do nothing if @p index is not valid.
     *
     * It doesn't invalidate any iterators which not point to deleted element.
     * @param index index of element to remove
     */
    void erase(std::size_t index) {
        erase(getIteratorForIndex(index));
    }
};

}   // namespace plask

#endif // PLASK__BOUNDARY_CONDITIONS_H
