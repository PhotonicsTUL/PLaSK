#ifndef BOUNDARY_H
#define BOUNDARY_H

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
Class @c Boundary\<MeshType\> stores boundary for mesh of type @c MeshType.
It has @c get method which return @c Boundary\<MeshType\>::WithMesh instanse for mesh given as parameter.
@c Boundary\<MeshType\>::WithMesh represent set of points (indexes of points in given mesh) and allow for:
- checking if it includes point with given index (@c includes method),
- iterate over represented indexes (has begin() and end() methods).

Typically, you should call @c MeshType static methods to obtain value for @c Boundary\<MeshType\>.

Example:
@code
using namespace plask;
Boundary<RectilinearMesh2d> boundary;   //stores boundary for mesh of type RectilinearMesh2d
boundary = RectilinearMesh2d::getLeftBoundary();
// now boundary represent condition which choose indexes of points on left boundary of any RectilinearMesh2d instance
//...
RectilinearMesh2d mesh;
//... (add some points to mesh)
Boundary<RectilinearMesh2d>::WithMesh bwm = boundary.get(mesh); //or boundary(mesh);
// bwm represent set of points indexes which lies on left boundary of mesh
std::cout << "Does point with index 0 lies on left boundary? Answare: " << bwm.includes(0) << std::endl;

for (std::size_t index: bwm) {  //iterate over boundary points (indexes)
    std::cout << "Point with index " << index
              << " lies on left boundary and has coordinates: "
              << mesh[index] << std::endl;
}
@endcode

@section boundaries_impl Boundaries implementations.
Instance of @c Boundary\<MeshType\> in fact is only a holder which includes pointer to abstract class @c BoundaryImpl\<MeshType\>.
It points to subclass of @c BoundaryImpl\<MeshType\> which implements all boundary logic (all calls of @c Boundary\<MeshType\>::WithMesh methods are delegete to it).

So, writing new boundary for given type of mesh @c MeshType is writing subclass of @c BoundaryImpl\<MeshType\>.

Plask includes some @c BoundaryImpl\<MeshType\> implementation:
- @c PredicateBoundary\<MeshType\> is implementation which holds and uses predicate (given in constructor) to check which points lies on boundary.

*/

#include "../utils/iterators.h"
#include "../memory.h"

#include "../utils/metaprog.h"   //for is_callable

namespace plask {

/**
 * Template of base class for boundaries of mesh with given type.
 * @tparam MeshType type of mesh
 */
template <typename MeshType>
struct BoundaryImpl {

    /// Base class for boundary iterator implementation.
    typedef PolymorphicForwardIteratorImpl<std::size_t, std::size_t> IteratorImpl;

    /// Boundy iterator type.
    typedef PolymorphicForwardIterator<IteratorImpl> Iterator;

    /// iterator over indexes of mesh
    typedef Iterator const_iterator;
    typedef const_iterator iterator;

    virtual ~BoundaryImpl() {}

    virtual BoundaryImpl<MeshType>* clone() const;

    /**
     * Check if boundary includes point with given index.
     * @param mesh mesh
     * @param mesh_index valid index of point in @p mesh
     * @return @c true only if point with index @p mesh_index in @p mesh lies on boundary
     */
    virtual bool includes(const MeshType& mesh, std::size_t mesh_index) const = 0;

    /**
     * Get begin iterator over boundary points (which are defined by indexes in @p mesh).
     * @param mesh mesh
     * @return begin iterator over boundary points
     */
    virtual const_iterator begin(const MeshType& mesh) const = 0;

    /**
     * Get end iterator over boundary points (which are defined by indexes in @p mesh).
     * @param mesh mesh
     * @return end iterator over boundary points
     */
    virtual const_iterator end(const MeshType& mesh) const = 0;

    /**
     * Thin wrapper over boundary and mesh pair.
     *
     * Each its method call boundary method with the same name passing wrapped mesh as first argument.
     */
    struct WithMesh {

        typedef BoundaryImpl<MeshType>::const_iterator const_iterator;
        typedef BoundaryImpl<MeshType>::iterator iterator;

        const BoundaryImpl& boundary;
        const MeshType& mesh;

        WithMesh(const BoundaryImpl& boundary, const MeshType& mesh)
            : boundary(boundary), mesh(mesh) {}

        bool includes(std::size_t mesh_index) const {
            return boundary.includes(mesh, mesh_index);
        }

        const_iterator begin() const {
            return boundary.begin(mesh);
        }

        const_iterator end() const {
            return boundary.end(mesh);
        }

    };

    /**
     * Get boundary with mesh wrapper.
     * It allow for easier calling of this boundaring methods.
     * @param mesh mesh which should be passed by returnet object as first parameter for each calling to boundaries method
     * @return wrapper for @c this boundary and given @p mesh
     */
    WithMesh get(const MeshType& mesh) const {
        return WithMesh(*this, mesh);
    }

    /// Base class for boundary iterator implementation which includes reference to boundary.
    struct IteratorWithMeshImpl: public IteratorImpl {

        WithMesh& boundaryWithMesh;

        const BoundaryImpl& getBoundary() const { return boundaryWithMesh.boundary; }
        const MeshType& getMesh() const { return boundaryWithMesh.mesh; }

        IteratorWithMeshImpl(const BoundaryImpl& boundary, const MeshType& mesh):
            boundaryWithMesh(boundary, mesh) {}
    };

};

template <typename MeshType>
struct Boundary: public Holder< const BoundaryImpl<MeshType> > {

    typedef typename BoundaryImpl<MeshType>::WithMesh WithMesh;

    Boundary(const BoundaryImpl<MeshType>* to_hold = nullptr): Holder< const BoundaryImpl<MeshType> >(to_hold) {}

    WithMesh operator()(const MeshType& mesh) const { return this->holded->get(mesh); }
    WithMesh get(const MeshType& mesh) const { return this->holded->get(mesh); }
};

/**
 * Boundary which wrap and use predicate.
 * @tparam MeshType type of mesh
 * @tparam Predicate predicate which check if given point is in boundary, predicate can has exactly one of the following arguments set:
 * - MeshType::LocaLCoords coords (plask::vec over MeshType space)
 * - MeshType mesh, std::size_t index (mesh and index in mesh)
 * - std::size_t index, MeshType mesh (mesh and index in mesh)
 * - std::size_t index (index in mesh)
 */
template <typename MeshType, typename Predicate>
struct PredicateBoundary: public BoundaryImpl<MeshType> {

    struct PredicateIteratorImpl: public BoundaryImpl<MeshType>::IteratorWithMeshImpl {

        using BoundaryImpl<MeshType>::IteratorWithMeshImpl::getMesh;

        decltype(std::begin(getMesh())) meshIterator;
        decltype(std::end(getMesh())) meshIteratorEnd;

        PredicateIteratorImpl(const BoundaryImpl<MeshType>& boundary, const MeshType& mesh,
                              decltype(std::begin(mesh)) meshIterator):
            BoundaryImpl<MeshType>::IteratorWithMeshImpl(boundary, mesh),
            meshIterator(meshIterator),
            meshIteratorEnd(std::end(mesh)) {}

        virtual std::size_t dereference() const {
            return meshIterator.getIndex();
        }

      private:
        bool check_predicate(typename std::enable_if<is_callable<Predicate, typename MeshType::LocalCoords>::value >::type* = 0) {
            return static_cast<PredicateBoundary&>(this->getBoundary()).predicate(*meshIterator);
        }
        bool check_predicate(typename std::enable_if<is_callable<Predicate, MeshType, std::size_t>::value >::type* = 0) {
            return static_cast<PredicateBoundary&>(this->getBoundary()).predicate(getMesh(), meshIterator->getIndex());
        }
        bool check_predicate(typename std::enable_if<is_callable<Predicate, std::size_t, MeshType>::value >::type* = 0) {
            return static_cast<PredicateBoundary&>(this->getBoundary()).predicate(meshIterator->getIndex(), getMesh());
        }
        bool check_predicate(typename std::enable_if<is_callable<Predicate, std::size_t>::value >::type* = 0) {
            return static_cast<PredicateBoundary&>(this->getBoundary()).predicate(meshIterator->getIndex());
        }

      public:

        virtual void increment() {
            do {
                ++meshIterator;
            } while (meshIterator != meshIteratorEnd && check_predicate());
        }

        virtual bool equal(const typename BoundaryImpl<MeshType>::IteratorImpl& other) const {
            return meshIterator == static_cast<const PredicateIteratorImpl&>(other).meshIterator;
        }

        virtual typename BoundaryImpl<MeshType>::IteratorImpl* clone() const {
            return new PredicateIteratorImpl(*this);
        }

    };

    Predicate predicate;

    PredicateBoundary(Predicate predicate): predicate(predicate) {}

    virtual PredicateBoundary<MeshType, Predicate>* clone() const { return new PredicateBoundary<MeshType, Predicate>(predicate); }

private:
    bool check_predicate(const MeshType& mesh, std::size_t mesh_index, typename std::enable_if<is_callable<Predicate, typename MeshType::LocalCoords>::value >::type* = 0) {
        return predicate(mesh[mesh_index]);
    }
    bool check_predicate(const MeshType& mesh, std::size_t mesh_index, typename std::enable_if<is_callable<Predicate, MeshType, std::size_t>::value >::type* = 0) {
        return predicate(mesh, mesh_index);
    }
    bool check_predicate(const MeshType& mesh, std::size_t mesh_index, typename std::enable_if<is_callable<Predicate, std::size_t, MeshType>::value >::type* = 0) {
        return predicate(mesh_index, mesh);
    }
    bool check_predicate(const MeshType&, std::size_t mesh_index, typename std::enable_if<is_callable<Predicate, std::size_t>::value >::type* = 0) {
        return predicate(mesh_index);
    }

public:

    virtual bool includes(const MeshType& mesh, std::size_t mesh_index) const {
        return check_predicate(mesh, mesh_index);
    }

    typename BoundaryImpl<MeshType>::Iterator begin(const MeshType &mesh) const {
        return typename BoundaryImpl<MeshType>::Iterator(new PredicateIteratorImpl(*this, mesh, std::begin(mesh)));
    }

    typename BoundaryImpl<MeshType>::Iterator end(const MeshType &mesh) const {
        return typename BoundaryImpl<MeshType>::Iterator(new PredicateIteratorImpl(*this, mesh, std::end(mesh)));
    }

};

/**
 * Helper to create boundary which wrap predicate.
 * Use: makePredicateBoundary<MeshType>(predicate);
 * @param predicate
 * @return <code>Boundary<MeshType>(new PredicateBoundary<MeshType, Predicate>(predicate))</code>
 * @tparam MeshType type of mesh
 * @tparam Predicate predicate which check if given point is in boundary, predicate can has exactly one of the following arguments set:
 * - MeshType::LocaLCoords coords (plask::vec over MeshType space)
 * - MeshType mesh, std::size_t index (mesh and index in mesh)
 * - std::size_t index, MeshType mesh (mesh and index in mesh)
 * - std::size_t index (index in mesh)
 */
template <typename MeshType, typename Predicate>
inline Boundary<MeshType> makePredicateBoundary(Predicate predicate) {
    return Boundary<MeshType>(new PredicateBoundary<MeshType, Predicate>(predicate));
}

}   // namespace plask

#endif // BOUNDARY_H
