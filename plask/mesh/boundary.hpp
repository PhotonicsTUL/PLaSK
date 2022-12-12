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
It has @c get method which return @ref BoundaryNodeSet "BoundaryNodeSet" instance for mesh given as parameter.
@ref BoundaryNodeSet "BoundaryNodeSet" represent a set of points (indexes of points in given mesh) and allow for:
- checking if it contains point with given index (@ref BoundaryNodeSet::contains "contains" method),
- iterate over represented indexes (has @ref BoundaryNodeSet::begin "begin" and @ref BoundaryNodeSet::end "end" methods).

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

BoundaryNodeSet bwm = boundary.get(mesh); // or boundary(mesh);
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
Instance of @ref BoundaryNodeSet "BoundaryNodeSet" in fact is only a holder which contains pointer to abstract class @ref plask::BoundaryNodeSetImpl.
It points to subclass of @ref plask::BoundaryNodeSetImpl which implements all boundary logic (all calls of @ref plask::BoundaryNodeSet "BoundaryNodeSet" methods are delegete to it).

So, writing new boundary for given type of mesh @c MeshType is writing subclass of @ref plask::BoundaryNodeSetImpl.

PLaSK contains some universal @ref plask::BoundaryNodeSetImpl "BoundaryNodeSet" implementations:
- @ref plask::EmptyBoundaryImpl is implementation which represents empty set of indexes,
    You can construct boundary which use this logic using @ref plask::makeEmptyBoundary function,
- @ref plask::PredicateBoundaryImpl "PredicateBoundaryImpl\<MeshType, Predicate\>" is implementation which holds and uses predicate (given in constructor) to check which points lies on boundary,
    You can construct boundary which use this logic using @ref plask::makePredicateBoundary function.
*/

#include "../utils/iterators.hpp"
#include "../utils/xml/reader.hpp"
#include "../memory.hpp"
#include "../log/log.hpp"

#include "../exceptions.hpp"
#include "../geometry/space.hpp"
#include <vector>
#include <set>

namespace plask {

/**
 * Base class for boundaries logic. Reperesnt polymorphic set of mesh indexes.
 * @see @ref boundaries
 */
struct PLASK_API BoundaryNodeSetImpl {

    /// Base class for boundary iterator implementation.
    typedef PolymorphicForwardIteratorImpl<std::size_t, std::size_t> IteratorImpl;

    /// Boundary iterator type.
    typedef PolymorphicForwardIterator<IteratorImpl> Iterator;

    /// iterator over indexes of mesh
    typedef Iterator const_iterator;
    typedef const_iterator iterator;

    virtual ~BoundaryNodeSetImpl() {}

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
    virtual std::size_t size() const { return std::size_t(std::distance(begin(), end())); }

};

/**
 * Template of base class for boundaries of mesh with given type which store reference to mesh.
 * @tparam MeshType type of mesh
 * @ref boundaries
 */
template <typename MeshType>
struct BoundaryNodeSetWithMeshImpl: public BoundaryNodeSetImpl {

    /// iterator over indexes of mesh
    typedef typename BoundaryNodeSetImpl::const_iterator const_iterator;

    /// iterator over indexes of mesh
    typedef typename BoundaryNodeSetImpl::iterator iterator;

    /// Held mesh.
    const MeshType& mesh;

    /**
     * Construct object which holds reference to given @p mesh.
     * @param mesh mesh to hold
     */
    BoundaryNodeSetWithMeshImpl(const MeshType& mesh)
        : mesh(mesh) {}

    /**
     * Base class for boundary iterator implementation which contains reference to boundary.
     *
     * Should not be used directly, but can save some work when implementing own BoundaryImpl and Iterator.
     */
    struct IteratorWithMeshImpl: public BoundaryNodeSetImpl::IteratorImpl {

        const BoundaryNodeSetWithMeshImpl<MeshType>& boundaryWithMesh;

        const BoundaryNodeSetWithMeshImpl<MeshType>& getBoundary() const { return boundaryWithMesh; }
        const MeshType& getMesh() const { return boundaryWithMesh.mesh; }

        IteratorWithMeshImpl(const BoundaryNodeSetWithMeshImpl<MeshType>& boundaryImpl)
            : boundaryWithMesh(boundaryImpl) {}
    };

};

/**
 * Holds BoundaryNodeSetImpl and delegate all calls to it.
 */
struct PLASK_API BoundaryNodeSet: public HolderRef< const BoundaryNodeSetImpl > {

    typedef typename BoundaryNodeSetImpl::const_iterator const_iterator;
    typedef typename BoundaryNodeSetImpl::iterator iterator;

    /**
     * Construct a boundary which holds given boundary logic.
     * @param to_hold pointer to object which describe boundary logic
     */
    BoundaryNodeSet(const BoundaryNodeSetImpl* to_hold = nullptr): HolderRef< const BoundaryNodeSetImpl >(to_hold) {}

    virtual ~BoundaryNodeSet() {}

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
 * This boundary represents an empty set of indices.
 */
struct PLASK_API EmptyBoundaryImpl: public BoundaryNodeSetImpl {

    struct IteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

        std::size_t dereference() const override {
            throw Exception("Dereference of empty boundary iterator.");
        }

        void increment() override {}

        bool equal(const typename BoundaryNodeSetImpl::IteratorImpl& PLASK_UNUSED(other)) const override {
            return true;
        }

        std::unique_ptr<typename BoundaryNodeSetImpl::IteratorImpl> clone() const override {
            return std::unique_ptr<typename BoundaryNodeSetImpl::IteratorImpl>(new IteratorImpl);
        }

    };

    bool contains(std::size_t PLASK_UNUSED(mesh_index)) const override { return false; }

    typename BoundaryNodeSetImpl::const_iterator begin() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl);
    }

    typename BoundaryNodeSetImpl::const_iterator end() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl);
    }

    std::size_t size() const override {
        return 0;
    }

    bool empty() const override { return true; }
};

/**
 * Implementation of boundary logic which holds a set of node indices in std::set.
 * It serves the node indices in increasing order.
 */
struct PLASK_API StdSetBoundaryImpl: public BoundaryNodeSetImpl {

    typedef std::set<std::size_t> StdNodeSet;

    typedef PolymorphicForwardIteratorWrapperImpl<StdNodeSet::const_iterator, std::size_t, std::size_t> IteratorImpl;

    StdNodeSet set;

    StdSetBoundaryImpl(StdNodeSet set): set(std::move(set)) {}

    bool contains(std::size_t mesh_index) const override {
        return set.find(mesh_index) != set.end();
    }

    typename BoundaryNodeSetImpl::const_iterator begin() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl(set.begin()));
    }

    typename BoundaryNodeSetImpl::const_iterator end() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl(set.end()));
    }

    std::size_t size() const override {
        return set.size();
    }

    bool empty() const override {
        return set.empty();
    }
};

// TODO może wykluczyć MeshType jako parametr szablonu i dodać w zamian DIM (używać MeshD)?
/**
 * Instance of this class represents predicate which chooses a subset of points (strictly: indices of points) from a mesh.
 * The mesh must be a specific type @p MeshType.
 *
 * Technically, Boundary is a factory of BoundaryNodeSet (objects that hold a pointer to abstract class @c BoundaryNodeSetImpl).
 * @tparam MeshType type of mesh
 * @ref boundaries
 */
template <typename MeshT>
struct Boundary {

    typedef MeshT MeshType;

protected:
    std::function<BoundaryNodeSet(const MeshType&, const shared_ptr<const GeometryD<MeshType::DIM>>&)> create;

public:

    //template <typename... T>
    //Boundary(T&&... args): create(std::forward<T>(args)...) {}

    Boundary(std::function<BoundaryNodeSet(const MeshType&, const shared_ptr<const GeometryD<MeshType::DIM>>&)> create_fun): create(create_fun) {}

    /// Construct null boundary.
    Boundary() {}

    //Boundary(const Boundary<MeshType>&) = default;
    //Boundary(Boundary<MeshType>&&) = default;

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    BoundaryNodeSet operator()(const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        if (isNull()) return new EmptyBoundaryImpl();
        return this->create(mesh, geometry);
    }

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    BoundaryNodeSet operator()(const shared_ptr<const MeshType>& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        if (isNull()) return new EmptyBoundaryImpl();
        return this->create(*mesh, geometry);
    }

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    BoundaryNodeSet get(const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
        if (isNull()) return new EmptyBoundaryImpl();
        return this->create(mesh, geometry);
    }

    /**
     * Get boundary-mesh pair for this boundary and given @p mesh.
     * @param mesh mesh
     * @return wrapper for @c this boundary and given @p mesh, it is valid only to time when both @p mesh and @c this are valid (not deleted)
     */
    BoundaryNodeSet get(const shared_ptr<const MeshType>& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) const {
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

/// Boundary which represents union/intersection/difference (depending on OpNodeSetImplT) of boundaries.
template <typename MeshType, typename OpNodeSetImplT>
struct BoundaryOp {

    Boundary<MeshType> A, B;

    BoundaryOp(Boundary<MeshType> A, Boundary<MeshType> B): A(std::move(A)), B(std::move(B)) {}

    BoundaryNodeSet operator()(const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geom) const {
        return new OpNodeSetImplT(A.get(mesh, geom), B.get(mesh, geom));
    }
};

/**
 * This logic holds a list of boundaries and represent a set of indices which is an union of sets from this boundaries.
 */
struct UnionBoundarySetImpl: public BoundaryNodeSetImpl {

    typedef std::vector< BoundaryNodeSet > BoundariesVec;
    BoundariesVec boundaries;

    struct IteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

        struct IteratorWithEnd {
            BoundaryNodeSet::const_iterator iter;
            BoundaryNodeSet::const_iterator end;

            IteratorWithEnd(BoundaryNodeSet::const_iterator iter, BoundaryNodeSet::const_iterator end)
                : iter(std::move(iter)), end(std::move(end)) {}

            bool valid() const { return iter != end; }

            bool operator==(const IteratorWithEnd& o) const { return iter == o.iter; }
        };

        std::vector<IteratorWithEnd> position;

        bool equal(const typename BoundaryNodeSetImpl::IteratorImpl &other) const override {
            const IteratorImpl& o = static_cast<const IteratorImpl&>(other);
            return position.size() == o.position.size() && std::equal(position.begin(), position.end(), o.position.begin());
        }

        std::unique_ptr<BoundaryNodeSetImpl::IteratorImpl> clone() const override {
            return std::unique_ptr<BoundaryNodeSetImpl::IteratorImpl>(new IteratorImpl(*this));
        }

    private:
        std::size_t minimal_position() const {
            std::size_t minimum = std::numeric_limits<std::size_t>::max();
            for (const IteratorWithEnd& v: position)
                if (v.valid()) {
                    auto vv = *v.iter;
                    if (vv < minimum) minimum = vv;
                }
            return minimum;
        }

    public:
        std::size_t dereference() const override {
            return minimal_position();
        }

        void increment() override {
            std::size_t m = minimal_position();
            for (IteratorWithEnd& v: position)
                if (v.valid() && *v.iter == m) ++v.iter;
        }

    };

    template <typename... Args>
    UnionBoundarySetImpl(Args&&... args):
        boundaries(std::forward<Args>(args)...) {   }

    UnionBoundarySetImpl(BoundaryNodeSet A, BoundaryNodeSet B)
        : boundaries(std::initializer_list<plask::BoundaryNodeSet>{A, B}) {}

    bool contains(std::size_t mesh_index) const override {
        for (auto& b: boundaries)
            if (b.contains(mesh_index)) return true;
        return false;
    }

    typename BoundaryNodeSetImpl::Iterator begin() const override {
        std::unique_ptr<IteratorImpl> impl(new IteratorImpl());
        impl->position.reserve(boundaries.size());
        for (const BoundaryNodeSet& b: boundaries)
            impl->position.emplace_back(b.begin(), b.end());
        return typename BoundaryNodeSetImpl::Iterator(impl.release());
    }

    typename BoundaryNodeSetImpl::Iterator end() const override {
        std::unique_ptr<IteratorImpl> impl(new IteratorImpl());
        impl->position.reserve(boundaries.size());
        for (const BoundaryNodeSet& b: boundaries)
            impl->position.emplace_back(b.end(), b.end());
        return typename BoundaryNodeSetImpl::Iterator(impl.release());
    }

    bool empty() const override {
        for (auto bound: boundaries)
            if (!bound.empty()) return false;
        return true;
    }

    /*std::size_t size() const override {
        std::size_t s = 0;
        for (auto bound: boundaries) s += bound.size();
        return s;
    }*/

    void push_back(const BoundaryNodeSet& to_append) { boundaries.push_back(to_append); }

    void push_back(BoundaryNodeSet&& to_append) { boundaries.push_back(std::move(to_append)); }

};

/**
 * This logic holds two boundaries and represent a set difference of them.
 */
struct DiffBoundarySetImpl: public BoundaryNodeSetImpl {

    BoundaryNodeSet A, B;

    struct IteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

        struct IteratorWithEnd {
            BoundaryNodeSet::const_iterator iter;
            BoundaryNodeSet::const_iterator end;

            IteratorWithEnd(BoundaryNodeSet::const_iterator iter, BoundaryNodeSet::const_iterator end)
                : iter(std::move(iter)), end(std::move(end)) {}

            bool valid() const { return iter != end; }

            bool operator==(const IteratorWithEnd& o) const { return iter == o.iter; }
        };

        IteratorWithEnd Apos, Bpos;

    private:
        void advanceAtoNearestValidPos() {
            while (Apos.valid()) {
                const std::size_t Aindex = *Apos.iter;
                while (true) {
                    if (!Bpos.valid()) return;  // accept current Apos
                    const std::size_t Bindex = *Bpos.iter;
                    if (Bindex == Aindex) break;    // go to next A index
                    if (Bindex > Aindex) return;    // accept current Apos
                    // here Bindex < Aindex
                    ++Bpos.iter;                         // go to next B index
                }
                ++Apos.iter;
            }
        }

    public:

        IteratorImpl(BoundaryNodeSet::const_iterator Aiter, BoundaryNodeSet::const_iterator Aend,
                     BoundaryNodeSet::const_iterator Biter, BoundaryNodeSet::const_iterator Bend):
            Apos(std::move(Aiter), std::move(Aend)), Bpos(std::move(Biter), std::move(Bend))
        {
            advanceAtoNearestValidPos();
        }

        bool equal(const typename BoundaryNodeSetImpl::IteratorImpl &other) const override {
            const IteratorImpl& o = static_cast<const IteratorImpl&>(other);
            return Apos == o.Apos;
        }

        std::unique_ptr<BoundaryNodeSetImpl::IteratorImpl> clone() const override {
            return std::unique_ptr<BoundaryNodeSetImpl::IteratorImpl>(new IteratorImpl(*this));
        }

        std::size_t dereference() const override {
            return *Apos.iter;
        }

        void increment() override {
            ++Apos.iter;
            advanceAtoNearestValidPos();
        }

    };

    DiffBoundarySetImpl(BoundaryNodeSet A, BoundaryNodeSet B):
        A(std::move(A)), B(std::move(B)) {   }

    bool contains(std::size_t mesh_index) const override {
        return A.contains(mesh_index) && !B.contains(mesh_index);
    }

    typename BoundaryNodeSetImpl::Iterator begin() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl(A.begin(), A.end(), B.begin(), B.end()));
    }

    typename BoundaryNodeSetImpl::Iterator end() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl(A.end(), A.end(), B.end(), B.end()));
    }

    bool empty() const override {
        return begin() == end();
    }

};


/**
 * This logic holds two boundaries and represent a set intersection of them.
 */
struct IntersectionBoundarySetImpl: public BoundaryNodeSetImpl {

    BoundaryNodeSet A, B;

    struct IteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

        struct IteratorWithEnd {
            BoundaryNodeSet::const_iterator iter;
            BoundaryNodeSet::const_iterator end;

            IteratorWithEnd(BoundaryNodeSet::const_iterator iter, BoundaryNodeSet::const_iterator end)
                : iter(std::move(iter)), end(std::move(end)) {}

            bool valid() const { return iter != end; }

            bool operator==(const IteratorWithEnd& o) const { return iter == o.iter; }
        };

        IteratorWithEnd Apos, Bpos;

    private:
        void advanceToNearestValidPos() {
            while (Apos.valid()) {
                if (!Bpos.valid()) {
                    Apos.iter = Apos.end;
                    return;
                }
                const std::size_t Aindex = *Apos.iter;
                const std::size_t Bindex = *Bpos.iter;
                if (Aindex == Bindex) return;
                if (Aindex < Bindex) ++Apos.iter; else ++Bpos.iter;
            }
        }

    public:

        IteratorImpl(BoundaryNodeSet::const_iterator Aiter, BoundaryNodeSet::const_iterator Aend,
                     BoundaryNodeSet::const_iterator Biter, BoundaryNodeSet::const_iterator Bend):
            Apos(std::move(Aiter), std::move(Aend)), Bpos(std::move(Biter), std::move(Bend))
        {
            advanceToNearestValidPos();
        }

        bool equal(const typename BoundaryNodeSetImpl::IteratorImpl &other) const override {
            const IteratorImpl& o = static_cast<const IteratorImpl&>(other);
            return Apos == o.Apos;
        }

        std::unique_ptr<BoundaryNodeSetImpl::IteratorImpl> clone() const override {
            return std::unique_ptr<BoundaryNodeSetImpl::IteratorImpl>(new IteratorImpl(*this));
        }

        std::size_t dereference() const override {
            return *Apos.iter;
        }

        void increment() override {
            ++Apos.iter;
            ++Bpos.iter;
            advanceToNearestValidPos();
        }

    };

    IntersectionBoundarySetImpl(BoundaryNodeSet A, BoundaryNodeSet B):
        A(std::move(A)), B(std::move(B)) {   }

    bool contains(std::size_t mesh_index) const override {
        return A.contains(mesh_index) && B.contains(mesh_index);
    }

    typename BoundaryNodeSetImpl::Iterator begin() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl(A.begin(), A.end(), B.begin(), B.end()));
    }

    typename BoundaryNodeSetImpl::Iterator end() const override {
        return typename BoundaryNodeSetImpl::Iterator(new IteratorImpl(A.end(), A.end(), B.end(), B.end()));
    }

    bool empty() const override {
        return begin() == end();
    }

};

/// Boundary which represents union of boundaries.
template <typename MeshType> using UnionBoundary = BoundaryOp<MeshType, UnionBoundarySetImpl>;

/// Boundary which represents difference of boundaries.
template <typename MeshType> using DiffBoundary = BoundaryOp<MeshType, DiffBoundarySetImpl>;

/// Boundary which represents intersection of boundaries.
template <typename MeshType> using IntersectionBoundary = BoundaryOp<MeshType, IntersectionBoundarySetImpl>;

//@{
/**
 * Return union of left and right sets of nodes.
 * @param left, right sets of nodes
 * @return the union
 */
inline BoundaryNodeSet operator+(BoundaryNodeSet left, BoundaryNodeSet right) {
    return new UnionBoundarySetImpl(std::move(left), std::move(right));
}
inline BoundaryNodeSet operator|(BoundaryNodeSet left, BoundaryNodeSet right) {
    return new UnionBoundarySetImpl(std::move(left), std::move(right));
}
//@}

//@{
/**
 * Return intersection of left and right sets of nodes.
 * @param left, right sets of nodes
 * @return the intersection
 */
inline BoundaryNodeSet operator*(BoundaryNodeSet left, BoundaryNodeSet right) {
    return new IntersectionBoundarySetImpl(std::move(left), std::move(right));
}
inline BoundaryNodeSet operator&(BoundaryNodeSet left, BoundaryNodeSet right) {
    return new IntersectionBoundarySetImpl(std::move(left), std::move(right));
}
//@}

/**
 * Return difference of left and right sets of nodes.
 * @param left, right sets of nodes
 * @return the difference
 */
inline BoundaryNodeSet operator-(BoundaryNodeSet left, BoundaryNodeSet right) {
    return new DiffBoundarySetImpl(std::move(left), std::move(right));
}

//@{
/**
 * Return boundary which produces the union of sets given by left and right boundaries.
 * @param left, right boundaries
 * @return the union boundary
 */
template <typename MeshType>
inline Boundary<MeshType> operator+(Boundary<MeshType> left, Boundary<MeshType> right) {
    return Boundary<MeshType>(UnionBoundary<MeshType>(std::move(left), std::move(right)));
}
template <typename MeshType>
inline Boundary<MeshType> operator|(Boundary<MeshType> left, Boundary<MeshType> right) {
    return Boundary<MeshType>(UnionBoundary<MeshType>(std::move(left), std::move(right)));
}
//@}

//@{
/**
 * Return boundary which produces the intersection of sets given by left and right boundaries.
 * @param left, right boundaries
 * @return the intersection boundary
 */
template <typename MeshType>
inline Boundary<MeshType> operator*(Boundary<MeshType> left, Boundary<MeshType> right) {
    return Boundary<MeshType>(IntersectionBoundary<MeshType>(std::move(left), std::move(right)));
}
template <typename MeshType>
inline Boundary<MeshType> operator&(Boundary<MeshType> left, Boundary<MeshType> right) {
    return Boundary<MeshType>(IntersectionBoundary<MeshType>(std::move(left), std::move(right)));
}
//@}

/**
 * Return boundary which produces the difference of sets given by left and right boundaries.
 * @param left, right boundaries
 * @return the difference boundary
 */
template <typename MeshType>
inline Boundary<MeshType> operator-(Boundary<MeshType> left, Boundary<MeshType> right) {
    return Boundary<MeshType>(DiffBoundary<MeshType>(std::move(left), std::move(right)));
}

/**
 * Boundary logic implementation which represents set of indexes which fulfill predicate.
 * @tparam MeshT type of mesh
 * @tparam predicate functor which check if given point is in boundary
 * @ref boundaries
 */
template <typename MeshT, typename Predicate>
struct PredicateBoundaryImpl: public BoundaryNodeSetWithMeshImpl<typename MeshT::Boundary::MeshType> {

    typedef typename MeshT::Boundary::MeshType MeshType;

    struct PredicateIteratorImpl: public BoundaryNodeSetWithMeshImpl<MeshType>::IteratorWithMeshImpl {

       typedef decltype(std::begin(std::declval<MeshType>())) MeshBeginIterator;
       typedef decltype(std::end(std::declval<MeshType>())) MeshEndIterator;

       MeshBeginIterator meshIterator;
       MeshEndIterator meshIteratorEnd;

       PredicateIteratorImpl(const BoundaryNodeSetWithMeshImpl<MeshType>& boundary, MeshBeginIterator meshIterator):
            BoundaryNodeSetWithMeshImpl<MeshType>::IteratorWithMeshImpl(boundary),
            meshIterator(meshIterator),
            meshIteratorEnd(std::end(boundary.mesh)) {
           while (this->meshIterator != meshIteratorEnd && !check_predicate())
               ++this->meshIterator;  //go to first element which fulfill predicate
       }

        std::size_t dereference() const override {
            return meshIterator.getIndex();
        }

      private:
        bool check_predicate() {
            return static_cast<const PredicateBoundaryImpl&>(this->getBoundary())
                .predicate(this->getMesh(), meshIterator.getIndex());
        }

      public:

        void increment() override {
            do {
                ++meshIterator;
            } while (meshIterator != meshIteratorEnd && !check_predicate());
        }

        bool equal(const typename BoundaryNodeSetImpl::IteratorImpl& other) const override {
            return meshIterator == static_cast<const PredicateIteratorImpl&>(other).meshIterator;
        }

        std::unique_ptr<typename BoundaryNodeSetImpl::IteratorImpl> clone() const override {
            return std::unique_ptr<typename BoundaryNodeSetImpl::IteratorImpl>(new PredicateIteratorImpl(*this));
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
        BoundaryNodeSetWithMeshImpl<MeshType>(mesh), predicate(predicate) {}

    //virtual PredicateBoundary<MeshType, Predicate>* clone() const { return new PredicateBoundary<MeshType, Predicate>(predicate); }

private:
    bool check_predicate(std::size_t mesh_index) const {
        return predicate(this->mesh, mesh_index);
    }

public:

    bool contains(std::size_t mesh_index) const override {
        return this->check_predicate(mesh_index);
    }

    typename BoundaryNodeSetImpl::Iterator begin() const override {
        return typename BoundaryNodeSetImpl::Iterator(new PredicateIteratorImpl(*this, std::begin(this->mesh)));
    }

    typename BoundaryNodeSetImpl::Iterator end() const override {
        return typename BoundaryNodeSetImpl::Iterator(new PredicateIteratorImpl(*this, std::end(this->mesh)));
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
 * Use: makePredicateBoundary\<MeshType::Boundary>(predicate);
 * @param predicate functor which check if given point is in boundary
 * @return boundary which represents set of indexes which fulfill @p predicate
 * @tparam Boundary type of boundary which should be returned
 * @tparam Predicate predicate which check if given point is in boundary, predicate has the following arguments:
 * - MeshType mesh (mesh and index in mesh), std::size_t index
 */
template <typename Boundary, typename Predicate>
inline Boundary makePredicateBoundary(Predicate predicate) {
    return Boundary( [=](const typename Boundary::MeshType& mesh, const shared_ptr<const GeometryD<Boundary::MeshType::DIM>>&) {
        return new PredicateBoundaryImpl<typename Boundary::MeshType, Predicate>(mesh, predicate);
    } );
}

struct Manager;


/**
 * Parse boundary of given type from string.
 *
 * Specialization of this function:
 * @code
 * template <> inline BoundaryType parseBoundary<BoundaryType>(const std::string& boundary_desc) { ... }
 * @endcode
 * are responsible to parse boundary of type @c BoundaryType from string.
 * @param boundary_desc boundary description, depends on type of boundary (which generaly depends on mesh type)
 * @param manager geometry manager
 * @return parsed boundary or Boundary() if can't parse given string
 */
template <typename Boundary>
inline Boundary parseBoundary(const std::string& PLASK_UNUSED(boundary_desc), Manager& PLASK_UNUSED(manager)) { return Boundary(); }


/**
 * Parse boundary from XML reader.
 *
 * It starts from tag which beginning is pointed by reader and (in case of successful parse) move reader to end of this tag.
 *
 * For given boundary type (Boundary) specialization of this function:
 * @code
 * template <> inline Boundary parseBoundary<Boundary>(XMLReader& boundary_desc) { ... }
 * @endcode
 * are responsible to parse boundary from XML for this boundary type (which generaly depends on mesh type).
 * @param boundary_desc boundary description, depends on type of mesh
 * @param manager geometry manager
 * @return parsed boundary or Boundary() if can't parse given tag
 */
template <typename Boundary>
inline Boundary parseBoundary(XMLReader& PLASK_UNUSED(boundary_desc), Manager& PLASK_UNUSED(manager)) { return Boundary(); }

}   // namespace plask

#endif // PLASK__BOUNDARY_H
