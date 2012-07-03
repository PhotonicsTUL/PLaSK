#ifndef PLASK__BOUNDARY_H
#define PLASK__BOUNDARY_H

/** @file
This file includes templates of base classes for mesh's boundaries.
@see @ref meshes
*/

/** @page boundaries Boundaries
@section boundaries_about About boundaries
Boundaries represent some conditions which allow to choose a subset of points (strictly: indexes of points) from mesh.
Boundaries are typically used by modules to show points for boundaries conditions.

@section boundaries_use How to use boundaries?
Boundaries are specific for given type of mesh.
Class MeshT::Boundary (which in most cases is same as @ref plask::Boundary "Boundary\<MeshT\>") stores boundary for mesh of type @c MeshT.
It has @c get method which return @ref plask::Boundary::WithMesh "MeshT::Boundary::WithMesh" instance for mesh given as parameter.
@ref plask::Boundary::WithMesh "MeshT::Boundary::WithMesh" represent a set of points (indexes of points in given mesh) and allow for:
- checking if it includes point with given index (@ref plask::Boundary::WithMesh::includes "includes" method),
- iterate over represented indexes (has @ref plask::Boundary::WithMesh::begin "begin" and @ref plask::Boundary::WithMesh::end "end" methods).

Typically, you should call @c MeshT static methods to obtain value for @ref plask::Boundary "Boundary\<MeshT\>".

Example:
@code
using namespace plask;
RectilinearMesh2D::Boundary boundary;   // stores boundary for mesh of type RectilinearMesh2D
boundary = RectilinearMesh2D::getLeftBoundary();
// now boundary represents condition which choose indexes of points on left boundary of any RectilinearMesh2D instance
//...
RectilinearMesh2D mesh;
//... (add some points to mesh)
Boundary<RectilinearMesh2D>::WithMesh bwm = boundary.get(mesh); //or boundary(mesh);
// bwm represent set of points indexes which lies on left boundary of mesh
std::cout << "Does point with index 0 lies on left boundary? Answer: " << bwm.includes(0) << std::endl;

for (std::size_t index: bwm) {  //iterate over boundary points (indexes)
    std::cout << "Point with index " << index
              << " lies on left boundary and has coordinates: "
              << mesh[index] << std::endl;
}
@endcode

@section boundaries_modules How to use boundaries in modules?
Modules hold @ref plask::BoundaryCondition "boundary conditions" which are pairs of:
boundary (described by plask::Boundary) and condidion (description depends from type of condition, can be module specific).
Class plask::BoundaryConditions is container template of such pairs (it depends from both types: mesh and condition).
So, typically, modules have one or more public fields of type @ref plask::BoundaryConditions "BoundaryConditions\<MeshT, ConditionType>".
User of module can call this fields methods to @ref plask::BoundaryConditions::add "add" boundary condition, and module can iterate over this boundary conditions.

See also @ref modules_writing_details.

@section boundaries_impl Boundaries implementations.
Instance of @ref plask::Boundary "Boundary\<MeshT\>" in fact is only a holder which includes pointer to abstract class @ref plask::BoundaryImpl "BoundaryImpl\<MeshT\>".
It points to subclass of @ref plask::BoundaryImpl "BoundaryImpl\<MeshT\>" which implements all boundary logic (all calls of @ref plask::Boundary::WithMesh "Boundary\<MeshT\>::WithMesh" methods are delegete to it).

So, writing new boundary for given type of mesh @c MeshT is writing subclass of @ref plask::BoundaryImpl "BoundaryImpl\<MeshT\>".

PLaSK includes some universal @ref plask::BoundaryImpl "BoundaryImpl\<MeshT\>" implementation:
- @ref plask::PredicateBoundary "PredicateBoundary\<MeshT\>" is implementation which holds and uses predicate (given in constructor) to check which points lies on boundary.

*/

#include "../utils/iterators.h"
#include "../memory.h"

#include "../utils/metaprog.h"   //for is_callable

namespace plask {

/**
 * Template of base class for boundaries of mesh with given type.
 * @tparam MeshT type of mesh
 * @ref boundaries
 */
template <typename MeshT>
struct BoundaryImpl {

    /// Base class for boundary iterator implementation.
    typedef PolymorphicForwardIteratorImpl<std::size_t, std::size_t> IteratorImpl;

    /// Boundary iterator type.
    typedef PolymorphicForwardIterator<IteratorImpl> Iterator;

    /// iterator over indexes of mesh
    typedef Iterator const_iterator;
    typedef const_iterator iterator;

    virtual ~BoundaryImpl() {}

    //virtual BoundaryImpl<MeshT>* clone() const = 0;

    /**
     * Check if boundary includes point with given index.
     * @param mesh mesh
     * @param mesh_index valid index of point in @p mesh
     * @return @c true only if point with index @p mesh_index in @p mesh lies on boundary
     */
    virtual bool includes(const MeshT& mesh, std::size_t mesh_index) const = 0;

    /**
     * Get begin iterator over boundary points (which are defined by indexes in @p mesh).
     * @param mesh mesh
     * @return begin iterator over boundary points
     */
    virtual const_iterator begin(const MeshT& mesh) const = 0;

    /**
     * Get end iterator over boundary points (which are defined by indexes in @p mesh).
     * @param mesh mesh
     * @return end iterator over boundary points
     */
    virtual const_iterator end(const MeshT& mesh) const = 0;

    /**
     * Thin wrapper over boundary and mesh pair. It shows points described by boundary in particular mesh.
     *
     * Each its method call BoundaryImpl method with the same name passing wrapped mesh as first argument.
     */
    struct WithMesh {

        /// iterator over indexes of mesh
        typedef BoundaryImpl<MeshT>::const_iterator const_iterator;

        /// iterator over indexes of mesh
        typedef BoundaryImpl<MeshT>::iterator iterator;

        /// Logic of hold boundary.
        const BoundaryImpl<MeshT>& boundary;

        /// Hold mesh.
        const MeshT& mesh;

        /**
         * Construct object which holds given @p boundary and @p mesh.
         * @param boundary boundary to hold
         * @param mesh mesh to hold
         */
        WithMesh(const BoundaryImpl<MeshT>& boundary, const MeshT& mesh)
            : boundary(boundary), mesh(mesh) {}

        /**
         * Check if boundary includes point with given index.
         * @param mesh_index valid index of point in hold mesh
         * @return @c true only if point with index @p mesh_index in hold mesh lies on boundary
         */
        bool includes(std::size_t mesh_index) const {
            return boundary.includes(mesh, mesh_index);
        }

        /**
         * Get begin iterator over boundary points (which are defined by indexes in hold mesh).
         * @return begin iterator over boundary points
         */
        const_iterator begin() const {
            return boundary.begin(mesh);
        }

        /**
         * Get end iterator over boundary points (which are defined by indexes in hold mesh).
         * @return end iterator over boundary points
         */
        const_iterator end() const {
            return boundary.end(mesh);
        }

    };

    /**
     * Get boundary with mesh wrapper.
     * It allows for easier calling of this boundaring methods.
     * @param mesh mesh which should be passed by returnet object as first parameter for each calling to boundaries method
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    WithMesh get(const MeshT& mesh) const {
        return WithMesh(*this, mesh);
    }

    /**
     * Base class for boundary iterator implementation which includes reference to boundary.
     *
     * Should not be used directly, but can save some work when implementing own BoundaryImpl and Iterator.
     */
    struct IteratorWithMeshImpl: public IteratorImpl {

        WithMesh boundaryWithMesh;

        const BoundaryImpl<MeshT>& getBoundary() const { return boundaryWithMesh.boundary; }
        const MeshT& getMesh() const { return boundaryWithMesh.mesh; }

        IteratorWithMeshImpl(const BoundaryImpl<MeshT>& boundary, const MeshT& mesh)
            : boundaryWithMesh(boundary, mesh) {}
    };

};

/**
 * Instance of this class represents some conditions which allow to choose a subset of points (strictly: indexes of points) from mesh.
 * This mesh must be a specific type @p MeshT.
 *
 * In fact Boundary is only a holder which includes pointer to abstract class @c BoundaryImpl\<MeshT\> which implements boundary logic.
 * @tparam MeshT type of mesh
 * @ref boundaries
 */
template <typename MeshT>
struct Boundary: public HolderRef< const BoundaryImpl<MeshT> > {

    /// Type of the mesh for which this boundary is defined
    typedef MeshT MeshType;

    /// Type of boundary-mesh pair which shows points described by boundary in particular mesh.
    typedef typename BoundaryImpl<MeshT>::WithMesh WithMesh;

    /**
     * Construct a boundary which holds given boundary logic.
     * @param to_hold pointer to object which describe boundary logic
     */
    Boundary(const BoundaryImpl<MeshT>* to_hold = nullptr): HolderRef< const BoundaryImpl<MeshT> >(to_hold) {}

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    WithMesh operator()(const MeshT& mesh) const { return this->hold->get(mesh); }

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    WithMesh get(const MeshT& mesh) const { return this->hold->get(mesh); }

    /**
     * Check if boundary includes point with given index.
     * @param mesh mesh
     * @param mesh_index valid index of point in @p mesh
     * @return @c true only if point with index @p mesh_index in @p mesh lies on boundary
     */
    bool includes(const MeshT& mesh, std::size_t mesh_index) const {
        return this->hold->includes(mesh, mesh_index);
    }
};

/**
 * Boundary logic implementation which wrap and use predicate.
 * @tparam MeshT type of mesh
 * @tparam Predicate predicate which check if given point is in boundary, predicate can has exactly one of the following arguments set:
 * - MeshT::LocaLCoords coords (plask::vec over MeshT space)
 * - MeshT mesh, std::size_t index (mesh and index in mesh)
 * - std::size_t index, MeshT mesh (mesh and index in mesh)
 * - std::size_t index (index in mesh)
 * @ref boundaries
 */
template <typename MeshT, typename Predicate>
struct PredicateBoundary: public BoundaryImpl<MeshT> {

    struct PredicateIteratorImpl: public BoundaryImpl<MeshT>::IteratorWithMeshImpl {

        using BoundaryImpl<MeshT>::IteratorWithMeshImpl::getMesh;

        decltype(std::begin(getMesh())) meshIterator;
        decltype(std::end(getMesh())) meshIteratorEnd;
        //decltype(std::begin(typename MeshT::Boundary::MeshType())) meshIterator;
        //decltype(std::end(typename MeshT::Boundary::MeshType())) meshIteratorEnd;

        PredicateIteratorImpl(const BoundaryImpl<MeshT>& boundary, const MeshT& mesh,
                              decltype(std::begin(mesh)) meshIterator):
            BoundaryImpl<MeshT>::IteratorWithMeshImpl(boundary, mesh),
            meshIterator(meshIterator),
            meshIteratorEnd(std::end(mesh)) {}

        virtual std::size_t dereference() const {
            return meshIterator.getIndex();
        }

      private:
        bool check_predicate() {
            return static_cast<PredicateBoundary&>(this->getBoundary()).predicate(getMesh(), meshIterator->getIndex());
        }

      public:

        virtual void increment() {
            do {
                ++meshIterator;
            } while (meshIterator != meshIteratorEnd && check_predicate());
        }

        virtual bool equal(const typename BoundaryImpl<MeshT>::IteratorImpl& other) const {
            return meshIterator == static_cast<const PredicateIteratorImpl&>(other).meshIterator;
        }

        /*virtual typename BoundaryImpl<MeshT>::IteratorImpl* clone() const {
            return new PredicateIteratorImpl(*this);
        }*/

    };

    /// Predicate which check if given point is in boundary.
    Predicate predicate;

    /**
     * Construct predicate boundary which use given @p predicate.
     * @param predicate predicate which check if given point is in boundary
     */
    PredicateBoundary(Predicate predicate): predicate(predicate) {}

    //virtual PredicateBoundary<MeshT, Predicate>* clone() const { return new PredicateBoundary<MeshT, Predicate>(predicate); }

private:
    bool check_predicate(const MeshT& mesh, std::size_t mesh_index) {
        return predicate(mesh, mesh_index);
    }

public:

    virtual bool includes(const MeshT& mesh, std::size_t mesh_index) const {
        return this->check_predicate(mesh, mesh_index);
    }

    typename BoundaryImpl<MeshT>::Iterator begin(const MeshT &mesh) const {
        return typename BoundaryImpl<MeshT>::Iterator(new PredicateIteratorImpl(*this, mesh, std::begin(mesh)));
    }

    typename BoundaryImpl<MeshT>::Iterator end(const MeshT &mesh) const {
        return typename BoundaryImpl<MeshT>::Iterator(new PredicateIteratorImpl(*this, mesh, std::end(mesh)));
    }

};

/**
 * Helper to create boundary which wrap predicate.
 * Use: makePredicateBoundary\<MeshT>(predicate);
 * @param predicate functor which check if given point is in boundary
 * @return <code>Boundary<MeshT>(new PredicateBoundary<MeshT, Predicate>(predicate))</code>
 * @tparam MeshT type of mesh
 * @tparam Predicate predicate which check if given point is in boundary, predicate can has exactly one of the following arguments set:
 * - MeshT::LocaLCoords coords (plask::vec over MeshT space)
 * - MeshT mesh, std::size_t index (mesh and index in mesh)
 * - std::size_t index, MeshT mesh (mesh and index in mesh)
 * - std::size_t index (index in mesh)
 */
template <typename MeshT, typename Predicate>
inline typename MeshT::Boundary makePredicateBoundary(Predicate predicate) {
    return typename MeshT::Boundary(new PredicateBoundary<typename MeshT::Boundary::MeshType, Predicate>(predicate));
}

}   // namespace plask

#endif // PLASK__BOUNDARY_H
