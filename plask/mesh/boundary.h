#ifndef BOUNDARY_H
#define BOUNDARY_H

/** @file
This file includes templates of base classes for mesh's boundaries.
@see @ref meshes
*/

#include "../utils/iterators.h"
#include "../memory.h"

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
 * @tparam MeshType
 * @tparam Predicate preicate which check if given point (passed as plask::vec over MeshType space) is in boundary
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

        virtual void increment() {
            do {
                ++meshIterator;
            } while (meshIterator != meshIteratorEnd &&
                     static_cast<PredicateBoundary&>(this->getBoundary()).predicate(*meshIterator));
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

    virtual bool includes(const MeshType& mesh, std::size_t mesh_index) const {
        return predicate(mesh[mesh_index]);
    }

    typename BoundaryImpl<MeshType>::Iterator begin(const MeshType &mesh) const {
        return typename BoundaryImpl<MeshType>::Iterator(new PredicateIteratorImpl(*this, mesh, std::begin(mesh)));
    }

    typename BoundaryImpl<MeshType>::Iterator end(const MeshType &mesh) const {
        return typename BoundaryImpl<MeshType>::Iterator(new PredicateIteratorImpl(*this, mesh, std::end(mesh)));
    }

};

/**
 * Use: makePredicateBoundary<MeshType>(predicate);
 */
template <typename MeshType, typename Predicate>
inline Boundary<MeshType> makePredicateBoundary(Predicate predicate) {
    return Boundary<MeshType>(new PredicateBoundary<MeshType, Predicate>(predicate));
}

}   // namespace plask

#endif // BOUNDARY_H
