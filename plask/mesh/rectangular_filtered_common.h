#ifndef RECTANGULAR_FILTERED_COMMON_H
#define RECTANGULAR_FILTERED_COMMON_H

#include <functional>

#include "rectangular.h"
#include "../utils/numbers_set.h"


namespace plask {

/**
 * Common base class for RectangularFilteredMesh 2D and 3D.
 *
 * Do not use directly.
 */
template <int DIM>
class RectangularFilteredMeshBase: public MeshD<DIM> {

protected:

    const RectangularMesh<DIM>* rectangularMesh;

    typedef CompressedSetOfNumbers<std::uint32_t> Set;

    /// numbers of rectangularMesh indexes which are in the corners of the elements enabled
    Set nodes;

    /// numbers of enabled elements
    Set elements;

    /**
     * Used by interpolation.
     * @param axis
     * @param wrapped_point_coord
     * @param index_lo
     * @param index_hi
     */
    static void findIndexes(const MeshAxis& axis, double wrapped_point_coord, std::size_t& index_lo, std::size_t& index_hi) {
        index_hi = axis.findUpIndex(wrapped_point_coord);
        if (index_hi+1 == axis.size()) --index_hi;    // p.c0 == axis0->at(axis0->size()-1)
        assert(index_hi > 0);
        index_lo = index_hi - 1;
    }

    /**
     * Used by nearest neighbor interpolation.
     * @param p point coordinate such that axis.at(index_lo) <= p <= axis.at(index_hi)
     * @param axis
     * @param index_lo, index_hi indexes
     * @return either @p index_lo or @p index_hi, index which minimize |p - axis.at(index)|
     */
    static std::size_t nearest(double p, const MeshAxis& axis, std::size_t index_lo, std::size_t index_hi) {
        return p - axis.at(index_lo) <= axis.at(index_hi) - p ? index_lo : index_hi;
    }

public:

    using typename MeshD<DIM>::LocalCoords;

    /// Returned by some methods to signalize that element or node (with given index(es)) is not included in the mesh.
    enum:std::size_t { NOT_INCLUDED = Set::NOT_INCLUDED };

    RectangularFilteredMeshBase(const RectangularMesh<DIM>* rectangularMesh)
        : rectangularMesh(rectangularMesh) {}

    /**
     * Iterator over nodes coordinates.
     *
     * Iterator of this type is faster than IndexedIterator used by parent class,
     * as it has constant time dereference operation while at method has logarithmic time complexity.
     *
     * One can use:
     * - getIndex() method of the iterator to get index of the node,
     * - getNumber() method of the iterator to get index of the node in the wrapped mesh.
     */
    class const_iterator: public Set::ConstIteratorFacade<const_iterator, LocalCoords> {

        const RectangularFilteredMeshBase* mesh;

        LocalCoords dereference() const {
            return mesh->at(this->getNumber());
        }

    public:

        template <typename... CtorArgs>
        explicit const_iterator(const RectangularFilteredMeshBase* mesh, CtorArgs&&... ctorArgs)
            : mesh(&mesh), Set::ConstIteratorFacade<const_iterator, LocalCoords>(std::forward<CtorArgs>(ctorArgs)...) {}

        const Set& set() const { return mesh->nodes; }
    };

    /// Iterator over nodes coordinates. The same as const_iterator, since non-const iterators are not supported.
    typedef const_iterator iterator;

    const_iterator begin() const { return const_iterator(*this, 0, nodes.segments.begin()); }
    const_iterator end() const { return const_iterator(*this, size(), nodes.segments.end()); }

    LocalCoords at(std::size_t index) const override {
        return rectangularMesh->at(nodes.at(index));
    }

    std::size_t size() const override { return nodes.size(); }

    bool empty() const override { return nodes.empty(); }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0->size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
        return rectangularMesh->index0(nodes.at(mesh_index));
    }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0->size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return rectangularMesh->index1(nodes.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return rectangularMesh->majorIndex(nodes.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t minorIndex(std::size_t mesh_index) const {
        return rectangularMesh->minorIndex(nodes.at(mesh_index));
    }

};

}   // namespace plask

#endif // RECTANGULAR_FILTERED_COMMON_H
