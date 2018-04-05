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

    /**
     * Get number of elements (for FEM method) in the first direction.
     * @return number of elements in the full rectangular mesh in the first direction (axis0 direction).
     */
    std::size_t getElementsCount0() const {
        return rectangularMesh->getElementsCount0();
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in the full rectangular mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        return rectangularMesh->getElementsCount1();
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return elements.size();
    }

    /**
     * Convert mesh index of bottom left element corner to index of this element.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        return elements.indexOf(rectangularMesh->getElementIndexFromLowIndex(nodes.at(mesh_index_of_el_bottom_left)));
    }

    /**
     * Convert element index to mesh index of bottom-left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        return nodes.indexOf(rectangularMesh.getElementMeshLowIndex(elements.at(element_index)));
    }

    /**
     * Convert an element index to mesh indexes of bottom-left corner of the element.
     * @param element_index index of the element, from 0 to getElementsCount()-1
     * @return axis 0 and axis 1 indexes of mesh,
     * you can easy calculate rest indexes of element corner by adding 1 to returned coordinates
     */
    Vec<DIM, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        return rectangularMesh->getElementMeshLowIndexes(elements.at(element_index));
    }

    /**
     * Get an area of a given element.
     * @param element_index index of the element
     * @return the area of the element with given index
     */
    double getElementArea(std::size_t element_index) const {
        return rectangularMesh->getElementArea(elements.at(element_index));
    }

    /**
     * Get first coordinate of point in the center of an elements.
     * @param index0 index of the element (axis0 index)
     * @return first coordinate of the point in the center of the element
     */
    double getElementMidpoint0(std::size_t index0) const { return rectangularMesh->getElementMidpoint0(index0); }

    /**
     * Get second coordinate of point in the center of an elements.
     * @param index1 index of the element (axis1 index)
     * @return second coordinate of the point in the center of the element
     */
    double getElementMidpoint1(std::size_t index1) const { return rectangularMesh->getElementMidpoint1(index1); }

    /**
     * Get point in the center of an element.
     * @param element_index index of Elements
     * @return point in center of element with given index
     */
    Vec<2, double> getElementMidpoint(std::size_t element_index) const {
        return rectangularMesh->getElementMidpoint(elements.at(element_index));
    }

    /**
     * Get an element as a rectangle.
     * @param element_index index of the element
     * @return the element as a rectangle (box)
     */
    Box2D getElementBox(std::size_t element_index) const {
        return rectangularMesh->getElementBox(elements.at(element_index));
    }

};

}   // namespace plask

#endif // RECTANGULAR_FILTERED_COMMON_H
