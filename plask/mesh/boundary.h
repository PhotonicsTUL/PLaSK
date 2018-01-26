#ifndef PLASK__BOUNDARY_H
#define PLASK__BOUNDARY_H

/** @file
This file contains templates of base classes for mesh's boundaries.
@see @ref meshes
*/

/** @page boundaries Boundaries

@section boundaries_about About boundaries
Boundaries represent some conditions which allow to choose a subset of points (strictly: indexes of points) from mesh.
Boundaries are typically used by solvers to show points for boundaries conditions.

@section boundaries_use How to use boundaries?
Boundaries are specific for given type of mesh.
Class MeshType::Boundary (which in most cases is same as @ref plask::Boundary "Boundary\<MeshType\>") stores boundary for mesh of type @c MeshType.
It has @c get method which return @ref plask::Boundary::WithMesh "MeshType::Boundary::WithMesh" instance for mesh given as parameter.
@ref plask::Boundary::WithMesh "MeshType::Boundary::WithMesh" represent a set of points (indexes of points in given mesh) and allow for:
- checking if it contains point with given index (@ref plask::Boundary::WithMesh::contains "contains" method),
- iterate over represented indexes (has @ref plask::Boundary::WithMesh::begin "begin" and @ref plask::Boundary::WithMesh::end "end" methods).

Typically, you should call @c MeshType static methods to obtain value for @ref plask::Boundary "Boundary\<MeshType\>".

Example:
@code
using namespace plask;

RectilinearMesh2D::Boundary boundary;   // stores boundary for mesh of type RectilinearMesh2D
boundary = RectilinearMesh2D::getLeftBoundary();

// now boundary represents condition which choose indexes of points on left boundary of any RectilinearMesh2D instance

//...
RectilinearMesh2D mesh;
//... (add some points to mesh)

Boundary<RectilinearMesh2D>::WithMesh bwm = boundary.get(mesh); // or boundary(mesh);
// bwm represent set of points indexes which lies on left boundary of mesh

std::cout << "Does point with index 0 lies on left boundary? Answer: " << bwm.contains(0) << std::endl;

for (std::size_t index: bwm) {  // iterate over boundary points (indexes)
    std::cout << "Point with index " << index
              << " lies on left boundary and has coordinates: "
              << mesh[index] << std::endl;
}
@endcode

@section boundaries_solvers How to use boundaries in solvers?
Solvers hold @ref plask::BoundaryCondition "boundary conditions" which are pairs of:
boundary (described by plask::Boundary) and condidion (description depends from type of condition, can be solver specific).
Class plask::BoundaryConditions is container template of such pairs (it depends from both types: mesh and condition).
So, typically, solvers have one or more public fields of type @ref plask::BoundaryConditions "BoundaryConditions\<MeshType, ConditionType>".
User of solver can call this fields methods to @ref plask::BoundaryConditions::add "add" boundary condition, and solver can iterate over this boundary conditions.

See also @ref solvers_writing_details.

@section boundaries_impl Boundaries implementations.
Instance of @ref plask::Boundary::WithMesh "Boundary\<MeshType\>::WithMesh" in fact is only a holder which contains pointer to abstract class @ref plask::BoundaryLogicImpl.
It points to subclass of @ref plask::BoundaryLogicImpl which implements all boundary logic (all calls of @ref plask::Boundary::WithMesh "Boundary\<MeshType\>::WithMesh" methods are delegete to it).

So, writing new boundary for given type of mesh @c MeshType is writing subclass of @ref plask::BoundaryLogicImpl.

PLaSK contains some universal @ref plask::BoundaryLogicImpl "BoundaryLogicImpl\<MeshType\>" implementation:
- @ref plask::EmptyBoundaryImpl is implementation which represents empty set of indexes,
    You can construct boundary which use this logic using @ref plask::makeEmptyBoundary function,
- @ref plask::PredicateBoundaryImpl "PredicateBoundaryImpl\<MeshType, Predicate\>" is implementation which holds and uses predicate (given in constructor) to check which points lies on boundary,
    You can construct boundary which use this logic using @ref plask::makePredicateBoundary function.
*/

#include "../utils/iterators.h"
#include "../utils/xml/reader.h"
#include "../memory.h"
#include "../log/log.h"

#include "../exceptions.h"
#include "../geometry/space.h"
#include <vector>

namespace plask {

/**
 * Base class for boundaries logic. Reperesnt polymorphic set of mesh indexes.
 * @see @ref boundaries
 */
struct PLASK_API BoundaryLogicImpl {

    /// Base class for boundary iterator implementation.
    typedef PolymorphicForwardIteratorImpl<std::size_t, std::size_t> IteratorImpl;

    /// Boundary iterator type.
    typedef PolymorphicForwardIterator<IteratorImpl> Iterator;

    /// iterator over indexes of mesh
    typedef Iterator const_iterator;
    typedef const_iterator iterator;

    virtual ~BoundaryLogicImpl() {}

    /**
     * Check if boundary contains point with given index.
     * @param mesh_index valid index of point in @p mesh
     * @return @c true only if point with index @p mesh_index in @p mesh lies on boundary
     */
    virtual bool contains(std::size_t mesh_index) const = 0;

    /**
     * Get begin iterator over boundary points.
     * @return begin iterator over boundary points
     */
    virtual const_iterator begin() const = 0;

    /**
     * Get end iterator over boundary points.
     * @return end iterator over boundary points
     */
    virtual const_iterator end() const = 0;

    /**
     * Check if this represents empty set of indexes.
     * @return @c true only if this represents empty set of indexes
     */
    virtual bool empty() const { return begin() == end(); }

    /**
     * Get number of points in this boundary.
     *
     * Default implementation just use std::distance(begin(), end()) which iterates over all indexes and can be slow, so this is often reimplemented in subclasses.
     * \return number of points in this boundary
     */
    virtual std::size_t size() const { return std::distance(begin(), end()); }

};

/**
 * Template of base class for boundaries of mesh with given type which store reference to mesh.
 * @tparam MeshType type of mesh
 * @ref boundaries
 */
template <typename MeshType>
struct BoundaryWithMeshLogicImpl: public BoundaryLogicImpl {

    /// iterator over indexes of mesh
    typedef typename BoundaryLogicImpl::const_iterator const_iterator;

    /// iterator over indexes of mesh
    typedef typename BoundaryLogicImpl::iterator iterator;

    /// Held mesh.
    const MeshType& mesh;

    /**
     * Construct object which holds reference to given @p mesh.
     * @param mesh mesh to hold
     */
    BoundaryWithMeshLogicImpl(const MeshType& mesh)
        : mesh(mesh) {}

    /**
     * Base class for boundary iterator implementation which contains reference to boundary.
     *
     * Should not be used directly, but can save some work when implementing own BoundaryImpl and Iterator.
     */
    struct IteratorWithMeshImpl: public BoundaryLogicImpl::IteratorImpl {

        const BoundaryWithMeshLogicImpl<MeshType>& boundaryWithMesh;

        const BoundaryWithMeshLogicImpl<MeshType>& getBoundary() const { return boundaryWithMesh; }
        const MeshType& getMesh() const { return boundaryWithMesh.mesh; }

        IteratorWithMeshImpl(const BoundaryWithMeshLogicImpl<MeshType>& boundaryImpl)
            : boundaryWithMesh(boundaryImpl) {}
    };

};

/**
 * Holds BoundaryLogicImpl and delegate all calls to it.
 */
struct PLASK_API BoundaryWithMesh: public HolderRef< const BoundaryLogicImpl > {

    typedef typename BoundaryLogicImpl::const_iterator const_iterator;
    typedef typename BoundaryLogicImpl::iterator iterator;

    /**
     * Construct a boundary which holds given boundary logic.
     * @param to_hold pointer to object which describe boundary logic
     */
    BoundaryWithMesh(const BoundaryLogicImpl* to_hold = nullptr): HolderRef< const BoundaryLogicImpl >(to_hold) {}

    virtual ~BoundaryWithMesh() {}

    /**
     * Check if boundary contains point with given index.
     * @param mesh_index valid index of point in @p mesh
     * @return @c true only if point with index @p mesh_index in @p mesh lies on boundary
     */
    bool contains(std::size_t mesh_index) const {
        return this->held->contains(mesh_index);
    }

    /**
     * Get begin iterator over boundary points.
     * @return begin iterator over boundary points
     */
    const_iterator begin() const {
        return this->held->begin();
    }

    /**
     * Get end iterator over boundary points.
     * @return end iterator over boundary points
     */
    const_iterator end() const {
        return this->held->end();
    }

    /**
     * Get number of points in this boundary
     * \return number of points in this boundary
     */
    std::size_t size() const {
        return this->held->size();
    }

    /**
     * Check if boundary represents empty set of indexes.
     * @return @c true only if this represents empty set of indexes
     */
    virtual bool empty() const { return this->held->empty(); }

};

/**
 * Implementation of empty boundary logic.
 *
 * This boundary represents empty index set.
 */
struct PLASK_API EmptyBoundaryImpl: public BoundaryLogicImpl {

    struct IteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        virtual std::size_t dereference() const override {
            throw Exception("Dereference of empty boundary iterator.");
        }

        virtual void increment() override {}

        virtual bool equal(const typename BoundaryLogicImpl::IteratorImpl& /*other*/) const override {
            return true;
        }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const override {
            return new IteratorImpl;
        }

    };

    virtual bool contains(std::size_t /*mesh_index*/) const override { return false; }

    virtual typename BoundaryLogicImpl::const_iterator begin() const override {
        return typename BoundaryLogicImpl::Iterator(new IteratorImpl);
    }

    virtual typename BoundaryLogicImpl::const_iterator end() const override {
        return typename BoundaryLogicImpl::Iterator(new IteratorImpl);
    }

    std::size_t size() const override {
        return 0;
    }

    virtual bool empty() const override { return true; }
};

/**
 * Instance of this class represents some conditions which allow to choose a subset of points (strictly: indexes of points) from mesh.
 * This mesh must be a specific type @p MeshType.
 *
 * In fact Boundary is factory of Boundary::WithMesh which is a holder which contains pointer to abstract class @c BoundaryImpl\<MeshType\> which implements boundary logic.
 * @tparam MeshType type of mesh
 * @ref boundaries
 */
template <typename MeshType>
struct Boundary {

    typedef BoundaryWithMesh WithMesh;

protected:
    std::function<WithMesh(const MeshType&, const shared_ptr<const GeometryD<MeshType::DIM>>&)> create;

public:

    //template <typename... T>
    //Boundary(T&&... args): create(std::forward<T>(args)...) {}

    Boundary(std::function<WithMesh(const MeshType&, const shared_ptr<const GeometryD<MeshType::DIM>>&)> create_fun): create(create_fun) {}

    Boundary() {}

    //Boundary(const Boundary<MeshType>&) = default;
    //Boundary(Boundary<MeshType>&&) = default;

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    WithMesh operator()(const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        if (isNull()) return new EmptyBoundaryImpl();
        return this->create(mesh, geometry);
    }

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    WithMesh operator()(const shared_ptr<const MeshType>& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        if (isNull()) return new EmptyBoundaryImpl();
        return this->create(*mesh, geometry);
    }

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    WithMesh get(const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        if (isNull()) return new EmptyBoundaryImpl();
        return this->create(mesh, geometry);
    }

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    WithMesh get(const shared_ptr<const MeshType>& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        if (isNull()) return new EmptyBoundaryImpl();
        return this->create(*mesh, geometry);
    }

    /**
     * Check if boundary, for given @p mesh, represents empty set of indexes.
     * @param mesh mesh
     * @return @c true only if this represents empty set of indexes of given @p mesh
     */
    bool empty(const MeshType& mesh) const {
        if (isNull()) return true;
        return get(mesh).empty();
    }

    /**
     * Check if boundary is null (doesn't hold valid creator). Null boundary behaves simillar to empty one.
     * @return @c true only if boundary doesn't hold valid creator
     */
    bool isNull() const {
        return !create;
    }
};

/**
 * This logic holds a list of boundaries and represent a set of indexes which is a sum of sets from this boundaries.
 */
template <typename MeshType>
struct SumBoundaryImpl: public BoundaryLogicImpl {

    typedef std::vector< typename Boundary<MeshType>::WithMesh > BoundariesVec;
    BoundariesVec boundaries;

    struct IteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        typename BoundariesVec::const_iterator current_boundary;
        typename BoundariesVec::const_iterator current_boundary_end;
        typename Boundary<MeshType>::WithMesh::const_iterator in_boundary;
        typename Boundary<MeshType>::WithMesh::const_iterator in_boundary_end;

        // Skip empty or finished boundaries and advance to the next one
        void fixCurrentBoundary() {
            while (in_boundary == in_boundary_end) {
                ++current_boundary;
                if (current_boundary == current_boundary_end) return;
                in_boundary = current_boundary->begin();
                in_boundary_end = current_boundary->end();
            }
        }

        // Special version for begin
        IteratorImpl(typename BoundariesVec::const_iterator current_boundary, typename BoundariesVec::const_iterator current_boundary_end)
            : current_boundary(current_boundary), current_boundary_end(current_boundary_end), in_boundary(current_boundary->begin()), in_boundary_end(current_boundary->end()) {
            if (current_boundary != current_boundary_end) fixCurrentBoundary();
        }

        // Special version for end
        IteratorImpl(typename BoundariesVec::const_iterator current_boundary_end)
            : current_boundary(current_boundary_end), current_boundary_end(current_boundary_end)
        {}

        bool equal(const typename BoundaryLogicImpl::IteratorImpl &other) const override {
            const IteratorImpl& o = static_cast<const IteratorImpl&>(other);
            if (current_boundary != o.current_boundary) return false;   // other outer-loop boundaries
            //same outer-loop boundaries:
            if (current_boundary == current_boundary_end) return true;  // and both are ends
            return in_boundary == o.in_boundary;    // both are no ends, compare inner-loop iterators
        }

        virtual IteratorImpl* clone() const override {
            return new IteratorImpl(*this);
        }

        virtual std::size_t dereference() const override {
            return *in_boundary;
        }

        virtual void increment() override {
            if (current_boundary != current_boundary_end) {
                ++in_boundary;
                fixCurrentBoundary();
            }
        }

    };

    template <typename... Args>
    SumBoundaryImpl(Args&&... args):
        boundaries(std::forward<Args>(args)...) {   }

    virtual bool contains(std::size_t mesh_index) const override {
        for (auto& b: boundaries)
            if (b.contains(mesh_index)) return true;
        return false;
    }

    typename BoundaryLogicImpl::Iterator begin() const override {
        if (boundaries.empty()) // boundaries.begin() == boundaries.end()
            return typename BoundaryLogicImpl::Iterator(new IteratorImpl(boundaries.end()));
        return typename BoundaryLogicImpl::Iterator(new IteratorImpl(boundaries.begin(), boundaries.end()));
    }

    typename BoundaryLogicImpl::Iterator end() const override {
        return typename BoundaryLogicImpl::Iterator(new IteratorImpl(boundaries.end()));
    }

    bool empty() const override {
        for (auto bound: boundaries) if (!bound.empty()) return false;
        return true;
    }
    
    std::size_t size() const override {
        std::size_t s = 0;
        for (auto bound: boundaries) s += bound.size();
        return s;
    }

    void push_back(const typename Boundary<MeshType>::WithMesh& to_append) { boundaries.push_back(to_append); }

    void push_back(typename Boundary<MeshType>::WithMesh&& to_append) { boundaries.push_back(to_append); }

};

/**
 * Boundary logic implementation which represents set of indexes which fulfill predicate.
 * @tparam MeshType type of mesh
 * @tparam predicate functor which check if given point is in boundary
 * @ref boundaries
 */
template <typename MeshType, typename Predicate>
struct PredicateBoundaryImpl: public BoundaryWithMeshLogicImpl<MeshType> {

    struct PredicateIteratorImpl: public BoundaryWithMeshLogicImpl<MeshType>::IteratorWithMeshImpl {

       typedef decltype(std::begin(std::declval<MeshType>())) MeshBeginIterator;
       typedef decltype(std::end(std::declval<MeshType>())) MeshEndIterator;

       MeshBeginIterator meshIterator;
       MeshEndIterator meshIteratorEnd;

       PredicateIteratorImpl(const BoundaryWithMeshLogicImpl<MeshType>& boundary, MeshBeginIterator meshIterator):
            BoundaryWithMeshLogicImpl<MeshType>::IteratorWithMeshImpl(boundary),
            meshIterator(meshIterator),
            meshIteratorEnd(std::end(boundary.mesh)) {
           while (this->meshIterator != meshIteratorEnd && !check_predicate())
               ++this->meshIterator;  //go to first element which fulfill predicate
       }

        virtual std::size_t dereference() const override {
            return meshIterator.getIndex();
        }

      private:
        bool check_predicate() {
            return static_cast<const PredicateBoundaryImpl&>(this->getBoundary())
                .predicate(this->getMesh(), meshIterator.getIndex());
        }

      public:

        virtual void increment() override {
            do {
                ++meshIterator;
            } while (meshIterator != meshIteratorEnd && !check_predicate());
        }

        virtual bool equal(const typename BoundaryLogicImpl::IteratorImpl& other) const override {
            return meshIterator == static_cast<const PredicateIteratorImpl&>(other).meshIterator;
        }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const override {
            return new PredicateIteratorImpl(*this);
        }

    };

    /// Predicate which check if given point is in boundary.
    Predicate predicate;

    /**
     * Construct predicate boundary which use given @p predicate.
     * @param mesh mesh for which boundary should be calculated
     * @param predicate predicate which check if given point is in boundary
     */
    PredicateBoundaryImpl(const MeshType& mesh, Predicate predicate):
        BoundaryWithMeshLogicImpl<MeshType>(mesh), predicate(predicate) {}

    //virtual PredicateBoundary<MeshType, Predicate>* clone() const { return new PredicateBoundary<MeshType, Predicate>(predicate); }

private:
    bool check_predicate(std::size_t mesh_index) const {
        return predicate(this->mesh, mesh_index);
    }

public:

    virtual bool contains(std::size_t mesh_index) const override {
        return this->check_predicate(mesh_index);
    }

    typename BoundaryLogicImpl::Iterator begin() const override {
        return typename BoundaryLogicImpl::Iterator(new PredicateIteratorImpl(*this, std::begin(this->mesh)));
    }

    typename BoundaryLogicImpl::Iterator end() const override {
        return typename BoundaryLogicImpl::Iterator(new PredicateIteratorImpl(*this, std::end(this->mesh)));
    }

};

/**
 * Helper to create empty boundary.
 * @return empty boundary
 * @tparam MeshType type of mesh for which boundary should be returned
 */
template <typename MeshType>
inline typename MeshType::Boundary makeEmptyBoundary() {
    return typename MeshType::Boundary( 
        [](const MeshType&, const shared_ptr<const GeometryD<MeshType::DIM>>&) { return new EmptyBoundaryImpl(); } 
    );
}


/**
 * Helper to create boundary which represents set of indexes which fulfill given @p predicate.
 * Use: makePredicateBoundary\<MeshType>(predicate);
 * @param predicate functor which check if given point is in boundary
 * @return boundary which represents set of indexes which fulfill @p predicate
 * @tparam MeshType type of mesh for which boundary should be returned
 * @tparam Predicate predicate which check if given point is in boundary, predicate has the following arguments:
 * - MeshType mesh (mesh and index in mesh), std::size_t index
 */
template <typename MeshType, typename Predicate>
inline typename MeshType::Boundary makePredicateBoundary(Predicate predicate) {
    return typename MeshType::Boundary( [=](const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>&) {
        return new PredicateBoundaryImpl<MeshType, Predicate>(mesh, predicate);
    } );
}

struct Manager;


/**
 * Parse boundary from string.
 *
 * For given mesh type (MyMeshType) specialization of this function:
 * @code
 * template <> inline Boundary<MyMeshType> parseBoundary<MyMeshType>(const std::string& boundary_desc) { ... }
 * @endcode
 * are responsible to parse boundary from string for this mesh type.
 * @param boundary_desc boundary description, depends from type of mesh
 * @param manager geometry manager
 * @return parsed boundary or Boundary<MeshType>() if can't parse given string
 */
template <typename MeshType>
inline Boundary<MeshType> parseBoundary(const std::string& boundary_desc, Manager& manager) { return Boundary<MeshType>(); }


/**
 * Parse boundary from XML reader.
 *
 * It starts from tag which beginning is pointed by reader and (in case of successful parse) move reader to end of this tag.
 *
 * For given mesh type (MyMeshType) specialization of this function:
 * @code
 * template <> inline Boundary<MyMeshType> parseBoundary<MyMeshType>(XMLReader& boundary_desc) { ... }
 * @endcode
 * are responsible to parse boundary from XML for this mesh type.
 * @param boundary_desc boundary description, depends from type of mesh
 * @param manager geometry manager
 * @return parsed boundary or Boundary<MeshType>() if can't parse given tag
 */
template <typename MeshType>
inline Boundary<MeshType> parseBoundary(XMLReader& boundary_desc, Manager& manager) { return Boundary<MeshType>(); }

}   // namespace plask

#endif // PLASK__BOUNDARY_H
