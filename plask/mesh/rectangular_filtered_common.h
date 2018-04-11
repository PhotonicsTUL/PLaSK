#ifndef PLASK__RECTANGULAR_FILTERED_COMMON_H
#define PLASK__RECTANGULAR_FILTERED_COMMON_H

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

    //typedef CompressedSetOfNumbers<std::uint32_t> Set;
    typedef CompressedSetOfNumbers<std::size_t> Set;

    /// numbers of rectangularMesh indexes which are in the corners of the elements enabled
    Set nodesSet;

    /// numbers of enabled elements
    Set elementsSet;

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

    /**
     * Base class for Elements with common code for 2D and 3D.
     */
    template <typename FilteredMeshType>
    struct ElementsBase {

        using Element = typename FilteredMeshType::Element;

        /**
         * Iterator over elements.
         *
         * One can use:
         * - getIndex() method of the iterator to get index of the element,
         * - getNumber() method of the iterator to get index of the element in the wrapped mesh.
         */
        class const_iterator: public Set::ConstIteratorFacade<const_iterator, Element> {

            const FilteredMeshType* filteredMesh;

            Element dereference() const {
                return Element(filteredMesh, this->getIndex(), this->getNumber());
            }

        public:

            template <typename... CtorArgs>
            explicit const_iterator(const FilteredMeshType* filteredMesh, CtorArgs&&... ctorArgs)
                : filteredMesh(&filteredMesh), Set::ConstIteratorFacade<const_iterator, Element>(std::forward<CtorArgs>(ctorArgs)...) {}

            const Set& set() const { return filteredMesh->elementsSet; }
        };

        /// Iterator over elments. The same as const_iterator, since non-const iterators are not supported.
        typedef const_iterator iterator;

        const FilteredMeshType* filteredMesh;

        explicit ElementsBase(const FilteredMeshType& filteredMesh): filteredMesh(&filteredMesh) {}

        /**
         * Get number of elements.
         * @return number of elements
         */
        std::size_t size() const { return filteredMesh->getElementsCount(); }

        /// @return iterator referring to the first element
        const_iterator begin() const { return const_iterator(this, 0, filteredMesh->elementsSet.segments.begin()); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(this, size(), filteredMesh->elementsSet.segments.end()); }

        /**
         * Get @p i-th element.
         * @param i element index
         * @return @p i-th element
         */
        Element operator[](std::size_t i) const { return Element(*filteredMesh, i); }

    };  // struct Elements

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
    class PLASK_API const_iterator: public Set::ConstIteratorFacade<const_iterator, LocalCoords> {

        const RectangularFilteredMeshBase* mesh;

        LocalCoords dereference() const {
            return mesh->at(this->getNumber());
        }

    public:

        template <typename... CtorArgs>
        explicit const_iterator(const RectangularFilteredMeshBase& mesh, CtorArgs&&... ctorArgs)
            : mesh(&mesh), Set::ConstIteratorFacade<const_iterator, LocalCoords>(std::forward<CtorArgs>(ctorArgs)...) {}

        const Set& set() const { return mesh->nodesSet; }
    };

    /// Iterator over nodes coordinates. The same as const_iterator, since non-const iterators are not supported.
    typedef const_iterator iterator;

    const_iterator begin() const { return const_iterator(*this, 0, nodesSet.segments.begin()); }
    const_iterator end() const { return const_iterator(*this, size(), nodesSet.segments.end()); }

    LocalCoords at(std::size_t index) const override {
        return rectangularMesh->at(nodesSet.at(index));
    }

    std::size_t size() const override { return nodesSet.size(); }

    bool empty() const override { return nodesSet.empty(); }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0->size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
        return rectangularMesh->index0(nodesSet.at(mesh_index));
    }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0->size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return rectangularMesh->index1(nodesSet.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return rectangularMesh->majorIndex(nodesSet.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t minorIndex(std::size_t mesh_index) const {
        return rectangularMesh->minorIndex(nodesSet.at(mesh_index));
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
        return elementsSet.size();
    }

    /**
     * Convert mesh index of bottom left element corner to index of this element.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        return elementsSet.indexOf(rectangularMesh->getElementIndexFromLowIndex(nodesSet.at(mesh_index_of_el_bottom_left)));
    }

    /**
     * Convert element index to mesh index of bottom-left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        return nodesSet.indexOf(rectangularMesh->getElementMeshLowIndex(elementsSet.at(element_index)));
    }

    /**
     * Convert an element index to mesh indexes of bottom-left corner of the element.
     * @param element_index index of the element, from 0 to getElementsCount()-1
     * @return axis 0 and axis 1 indexes of mesh,
     * you can easy calculate rest indexes of element corner by adding 1 to returned coordinates
     */
    Vec<DIM, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        return rectangularMesh->getElementMeshLowIndexes(elementsSet.at(element_index));
    }

    /**
     * Get an area of a given element.
     * @param element_index index of the element
     * @return the area of the element with given index
     */
    double getElementArea(std::size_t element_index) const {
        return rectangularMesh->getElementArea(elementsSet.at(element_index));
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
        return rectangularMesh->getElementMidpoint(elementsSet.at(element_index));
    }

    /**
     * Get an element as a rectangle.
     * @param element_index index of the element
     * @return the element as a rectangle (box)
     */
    typename Primitive<DIM>::Box getElementBox(std::size_t element_index) const {
        return rectangularMesh->getElementBox(elementsSet.at(element_index));
    }

};

}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED_COMMON_H
