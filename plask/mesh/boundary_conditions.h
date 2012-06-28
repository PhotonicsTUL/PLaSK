#ifndef PLASK__BOUNDARY_CONDITIONS_H
#define PLASK__BOUNDARY_CONDITIONS_H

#include "boundary.h"
#include <list>

namespace plask {

/**
 * Set of boundary conditions for given mesh type and boundary condition description type.
 * @tparam MeshT type of mesh
 * @tparam ConditionT type which describe boundary condition
 * @ref boundaries
 */
template <typename MeshT, typename ConditionT>
struct BoundaryConditions
{
    typename MeshT MeshType;    ///< type of mesh
    typename ConditionT ConditionType;  ///< type which describe boundary condition

    /// One boundary-condition pair.
    struct Element {
        Boundary<MeshType> boundary;

        ConditionType condition;

        /**
         * Construct boundary-condition pair.
         * @param boundary boundary
         * @param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
         */
        template <typename... ConditionArgumentsTypes>
        Element(Boundary&& boundary, ConditionArgumentsTypes&&... conditionsArg)
            :boundary(std::forward<boundary>(boundary)),
             condition(std::forward<ConditionArgumentsTypes>(conditionsArg)...) {}


    };

private:
    typedef std::list<Element> elements_container_t;
    elements_container_t container;
public:
    typedef elements_container_t::iterator iterator;
    typedef elements_container_t::const_iterator const_iterator;

    iterator begin() { return container.begin(); }
    const_iterator begin() const { return container.begin(); }

    iterator end() { return container.end(); }
    const_iterator end() const { return container.end(); }

    /**
     * Add new boundary condidion to this.
     * @param element boundary condidion to add
     */
    void add(Element&& element) {
        container.push_back(std::forward<Element>(element));
    }

    /**
     * Add new boundary condidion to this.
     * @param boundary boundary
     * @param conditionsArg arguments for condition constructor, can be just one argument of ConditionType to use copy/move-constructor
     */
    template <typename... ConditionArgumentsTypes>
    void add(Boundary&& boundary, ConditionArgumentsTypes&&... conditionsArg) {
        container.emplace_back(std::forward<Boundary>(boundary), std::forward<ConditionArgumentsTypes>(conditionsArg)...);
    }

    /// Delete all boundaries conditions from this set.
    void clear() {
        container.clear();
    }
};

}   // namespace plask

#endif // PLASK__BOUNDARY_CONDITIONS_H
