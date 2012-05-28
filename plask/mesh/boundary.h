#ifndef BOUNDARY_H
#define BOUNDARY_H

/** @file
This file includes templates of base classes for mesh's boundaries.
@see @ref meshes
*/

#include "../utils/iterators.h"

namespace plask {

/**
 * Template of base class for boundaries of mesh with given type.
 * @tparam MeshType type of mesh
 */
template <typename MeshType>
struct Boundary {

    /// Base class for boundry iterator implementation.
    typedef PolymorphicForwardIteratorImpl<std::size_t, std::size_t> IteratorImpl;

    /// Boundy iterator type.
    typedef PolymorphicForwardIterator<IteratorImpl> Iterator;

    /// iterator over indexes of mesh
    typedef Iterator const_iterator;
    typedef const_iterator iterator;

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

        typedef Boundary<MeshType>::const_iterator const_iterator;
        typedef Boundary<MeshType>::iterator iterator;

        const Boundary& boundary;
        const MeshType& mesh;

        WithMesh(const Boundary& boundary, const MeshType& mesh)
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

    /// Base class for boundry iterator implementation which includes reference to boundry.
    struct IteratorWithMeshImpl: public IteratorImpl {

        WithMesh& boundaryWithMesh;

        const Boundary& getBoundary() const { return boundaryWithMesh.boundary; }
        const MeshType& getMesh() const { return boundaryWithMesh.mesh; }

        IteratorWithMeshImpl(const Boundary& boundary, const MeshType& mesh):
            boundaryWithMesh(boundary, mesh) {}
    };

};

/**
 * Boundary which wrap and use predicate.
 * @tparam MeshType
 * @tparam Predicate preicate which check if given point (passed as plask::vec over MeshType space) is in boundary
 */
template <typename MeshType, typename Predicate>
struct PredicateBoundry: public Boundary<MeshType> {

    struct PredicateIteratorImpl: public Boundary<MeshType>::IteratorWithMeshImpl {

        using Boundary<MeshType>::IteratorWithMeshImpl::getMesh;

        decltype(std::begin(getMesh())) meshIterator;
        decltype(std::end(getMesh())) meshIteratorEnd;

        PredicateIteratorImpl(const Boundary<MeshType>& boundary, const MeshType& mesh,
                              decltype(std::begin(mesh)) meshIterator):
            Boundary<MeshType>::IteratorWithMeshImpl(boundary, mesh),
            meshIterator(meshIterator),
            meshIteratorEnd(std::end(mesh)) {}

        virtual std::size_t dereference() const {
            return meshIterator.getIndex();
        }

        virtual void increment() {
            do {
                ++meshIterator;
            } while (meshIterator != meshIteratorEnd &&
                     static_cast<PredicateBoundry&>(this->getBoundary()).predicate(*meshIterator));
        }

        virtual bool equal(const typename Boundary<MeshType>::IteratorImpl& other) const {
            return meshIterator == static_cast<const PredicateIteratorImpl&>(other).meshIterator;
        }

        virtual typename Boundary<MeshType>::IteratorImpl* clone() const {
            return new PredicateIteratorImpl(*this);
        }

    };

    Predicate predicate;

    PredicateBoundry(Predicate predicate): predicate(predicate) {}

    virtual bool includes(const MeshType& mesh, std::size_t mesh_index) const {
        return predicate(mesh[mesh_index]);
    }

    typename Boundary<MeshType>::Iterator begin(const MeshType &mesh) const {
        return typename Boundary<MeshType>::Iterator(new PredicateIteratorImpl(*this, mesh, std::begin(mesh)));
    }

    typename Boundary<MeshType>::Iterator end(const MeshType &mesh) const {
        return typename Boundary<MeshType>::Iterator(new PredicateIteratorImpl(*this, mesh, std::end(mesh)));
    }

};

/**
 * Use: makePredicateBoundary<MeshType>(predicate);
 */
template <typename MeshType, typename Predicate>
inline PredicateBoundry<MeshType, Predicate> makePredicateBoundary(Predicate predicate) {
    return PredicateBoundry<MeshType, Predicate>(predicate);
}

}   // namespace plask

#endif // BOUNDARY_H
