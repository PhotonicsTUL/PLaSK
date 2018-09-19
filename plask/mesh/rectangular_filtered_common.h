#ifndef PLASK__RECTANGULAR_FILTERED_COMMON_H
#define PLASK__RECTANGULAR_FILTERED_COMMON_H

#include <functional>

#include "rectangular.h"
#include "../utils/numbers_set.h"

#include <boost/thread.hpp>


namespace plask {

/**
 * Common base class for RectangularFilteredMesh 2D and 3D.
 *
 * Do not use directly.
 */
template <int DIM>
struct RectangularFilteredMeshBase: public RectangularMeshBase<DIM> {

    /// Maximum distance from boundary to include in the inerpolation
    constexpr static double MIN_DISTANCE = 1e-6; // 1 picometer

    /// Full, rectangular, wrapped mesh.
    RectangularMesh<DIM> fullMesh;

protected:

    //typedef CompressedSetOfNumbers<std::uint32_t> Set;
    typedef CompressedSetOfNumbers<std::size_t> Set;

    /// Numbers of rectangularMesh indexes which are in the corners of the elements enabled.
    Set nodeSet;

    /// Numbers of enabled elements.
    Set elementSet;

    /// The lowest and the largest index in use, for each direction.
    struct { std::size_t lo, up; } boundaryIndex[DIM];

    /**
     * Used by interpolation.
     * @param axis
     * @param wrapped_point_coord
     * @param index_lo
     * @param index_hi
     */
    static void findIndexes(const MeshAxis& axis, double wrapped_point_coord, std::size_t& index_lo, std::size_t& index_hi) {
        index_hi = axis.findUpIndex(wrapped_point_coord);
        if (index_hi == axis.size()) --index_hi;    // p.c0 == axis0->at(axis0->size()-1)
        else if (index_hi == 0) index_hi = 1;
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
                return Element(*filteredMesh, this->getIndex(), this->getNumber());
            }

            friend class boost::iterator_core_access;

        public:

            template <typename... CtorArgs>
            explicit const_iterator(const FilteredMeshType& filteredMesh, CtorArgs&&... ctorArgs)
                : Set::ConstIteratorFacade<const_iterator, Element>(std::forward<CtorArgs>(ctorArgs)...), filteredMesh(&filteredMesh) {}

            const Set& set() const { return filteredMesh->elementSet; }
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
        const_iterator begin() const { return const_iterator(*filteredMesh, 0, filteredMesh->elementSet.segments.begin()); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(*filteredMesh, size(), filteredMesh->elementSet.segments.end()); }

        /**
         * Get @p i-th element.
         * @param i element index
         * @return @p i-th element
         */
        Element operator[](std::size_t i) const { return Element(*filteredMesh, i); }

    };  // struct Elements

    void resetBoundyIndex() {
        for (int d = 0; d < DIM; ++d) { // prepare for finding indexes by subclass constructor:
            boundaryIndex[d].lo = this->fullMesh.axis[d]->size()-1;
            boundaryIndex[d].up = 0;
        }
    }

    /// Clear nodeSet, elementSet and call resetBoundyIndex().
    void reset() {
        nodeSet.clear();
        elementSet.clear();
        resetBoundyIndex();
    }

public:

    using typename MeshD<DIM>::LocalCoords;

    /// Returned by some methods to signalize that element or node (with given index(es)) is not included in the mesh.
    enum:std::size_t { NOT_INCLUDED = Set::NOT_INCLUDED };

    /// Construct an empty mesh. One should use reset() method before using it.
    RectangularFilteredMeshBase() = default;

    /// Constructor which allows us to construct midpoints mesh.
    RectangularFilteredMeshBase(const RectangularMesh<DIM>& rectangularMesh, Set nodeSet, bool clone_axes = false)
        : fullMesh(rectangularMesh, clone_axes), nodeSet(std::move(nodeSet)), elementSetInitialized(false) {}

    /**
     * Construct a mesh by wrap of a given @p rectangularMesh.
     * @param rectangularMesh mesh to wrap (it is copied by the constructor)
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if true) or shared (if false; default)
     */
    RectangularFilteredMeshBase(const RectangularMesh<DIM>& rectangularMesh, bool clone_axes = false)
        : fullMesh(rectangularMesh, clone_axes) { resetBoundyIndex(); }

    /**
     * Iterator over nodes coordinates. It implements const_iterator for filtered meshes.
     *
     * Iterator of this type is faster than IndexedIterator used by parent class of filtered meshes,
     * as it has constant time dereference operation while <code>at</code> method has logarithmic time complexity.
     *
     * One can use:
     * - getIndex() method of the iterator to get index of the node,
     * - getNumber() method of the iterator to get index of the node in the wrapped mesh.
     */
    class const_iterator: public CompressedSetOfNumbers<std::size_t>::ConstIteratorFacade<const_iterator, LocalCoords> {

        friend class boost::iterator_core_access;

        const RectangularFilteredMeshBase* mesh;

        LocalCoords dereference() const {
            return mesh->fullMesh.at(this->getNumber());
        }

    public:

        template <typename... CtorArgs>
        explicit const_iterator(const RectangularFilteredMeshBase& mesh, CtorArgs&&... ctorArgs)
            : CompressedSetOfNumbers<std::size_t>::ConstIteratorFacade<const_iterator, LocalCoords>(std::forward<CtorArgs>(ctorArgs)...), mesh(&mesh) {}

        const CompressedSetOfNumbers<std::size_t>& set() const { return mesh->nodeSet; }
    };

    /// Iterator over nodes coordinates. The same as const_iterator, since non-const iterators are not supported.
    typedef const_iterator iterator;

    const_iterator begin() const { return const_iterator(*this, 0, nodeSet.segments.begin()); }
    const_iterator end() const { return const_iterator(*this, size(), nodeSet.segments.end()); }

    LocalCoords at(std::size_t index) const override {
        return fullMesh.at(nodeSet.at(index));
    }

    std::size_t size() const override { return nodeSet.size(); }

    bool empty() const override { return nodeSet.empty(); }

    /**
     * Calculate this mesh index using indexes of axis[0] and axis[1].
     * @param indexes index of axis[0] and axis[1]
     * @return this mesh index, from 0 to size()-1, or NOT_INCLUDED
     */
    inline std::size_t index(const Vec<DIM, std::size_t>& indexes) const {
        return nodeSet.indexOf(fullMesh.index(indexes));
    }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0->size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
        return fullMesh.index0(nodeSet.at(mesh_index));
    }

    /**
     * Calculate index of axis1 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis1, from 0 to axis1->size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return fullMesh.index1(nodeSet.at(mesh_index));
    }

    /**
     * Calculate indexes of axes.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return indexes of axes
     */
    inline Vec<DIM, std::size_t> indexes(std::size_t mesh_index) const {
        return fullMesh.indexes(nodeSet.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return fullMesh.majorIndex(nodeSet.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t minorIndex(std::size_t mesh_index) const {
        return fullMesh.minorIndex(nodeSet.at(mesh_index));
    }

    /**
     * Get number of elements (for FEM method) in the first direction.
     * @return number of elements in the full rectangular mesh in the first direction (axis0 direction).
     */
    std::size_t getElementsCount0() const {
        ensureHasElements();
        return fullMesh.getElementsCount0();
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in the full rectangular mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        ensureHasElements();
        return fullMesh.getElementsCount1();
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        ensureHasElements();
        return elementSet.size();
    }

    /**
     * Convert mesh index of bottom left element corner to index of this element.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        ensureHasElements();
        return elementSet.indexOf(fullMesh.getElementIndexFromLowIndex(nodeSet.at(mesh_index_of_el_bottom_left)));
    }

    /**
     * Convert element index to mesh index of bottom-left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        ensureHasElements();
        return nodeSet.indexOf(fullMesh.getElementMeshLowIndex(elementSet.at(element_index)));
    }

    /**
     * Convert an element index to mesh indexes of bottom-left corner of the element.
     * @param element_index index of the element, from 0 to getElementsCount()-1
     * @return axis 0 and axis 1 indexes of mesh,
     * you can easy calculate rest indexes of element corner by adding 1 to returned coordinates
     */
    Vec<DIM, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        ensureHasElements();
        return fullMesh.getElementMeshLowIndexes(elementSet.at(element_index));
    }

    /**
     * Get an area of a given element.
     * @param element_index index of the element
     * @return the area of the element with given index
     */
    double getElementArea(std::size_t element_index) const {
        ensureHasElements();
        return fullMesh.getElementArea(elementSet.at(element_index));
    }

    /**
     * Get first coordinate of point in the center of an elements.
     * @param index0 index of the element (axis0 index)
     * @return first coordinate of the point in the center of the element
     */
    double getElementMidpoint0(std::size_t index0) const {
        ensureHasElements();
        return fullMesh.getElementMidpoint0(index0);
    }

    /**
     * Get second coordinate of point in the center of an elements.
     * @param index1 index of the element (axis1 index)
     * @return second coordinate of the point in the center of the element
     */
    double getElementMidpoint1(std::size_t index1) const {
        ensureHasElements();
        return fullMesh.getElementMidpoint1(index1);
    }

    /**
     * Get point in the center of an element.
     * @param element_index index of Elements
     * @return point in center of element with given index
     */
    Vec<DIM, double> getElementMidpoint(std::size_t element_index) const {
        ensureHasElements();
        return fullMesh.getElementMidpoint(elementSet.at(element_index));
    }

    /**
     * Get an element as a rectangle.
     * @param element_index index of the element
     * @return the element as a rectangle (box)
     */
    typename Primitive<DIM>::Box getElementBox(std::size_t element_index) const {
        ensureHasElements();
        return fullMesh.getElementBox(elementSet.at(element_index));
    }

private:    // constructing elementSet from nodes set (element is chosen when all its vertices are chosen) on-deamand

    /// Only one thread can calculate elementSet
    DontCopyThisField<boost::mutex> writeElementSet;

    /// Whether elementSet is initialized (default for most contructors)
    bool elementSetInitialized = true;

    bool allVerticesIncluded(const RectangularMesh2D::Element& el) const {
        return nodeSet.includes(el.getLoLoIndex()) &&
               nodeSet.includes(el.getUpLoIndex()) &&
               nodeSet.includes(el.getLoUpIndex()) &&
               nodeSet.includes(el.getUpUpIndex());
    }

    bool allVerticesIncluded(const RectangularMesh3D::Element& el) {
        return nodeSet.includes(el.getLoLoLoIndex()) &&
               nodeSet.includes(el.getUpLoLoIndex()) &&
               nodeSet.includes(el.getLoUpLoIndex()) &&
               nodeSet.includes(el.getLoLoUpIndex()) &&
               nodeSet.includes(el.getLoUpUpIndex()) &&
               nodeSet.includes(el.getUpLoUpIndex()) &&
               nodeSet.includes(el.getUpUpLoIndex()) &&
               nodeSet.includes(el.getUpUpUpIndex());
    }

    void calculateElements() {
        boost::lock_guard<boost::mutex> lock((boost::mutex&)writeElementSet);
        if (elementSetInitialized) return;  // another thread has initilized elementSet just when we waited for mutex
        // TODO faster implementation
        for (auto el: fullMesh.elements())
            if (allVerticesIncluded(el)) elementSet.push_back(el.getIndex());
        elementSetInitialized = true;
    }

protected:

    /// Ensure that elementSet is calculated (calculate it if it is not)
    void ensureHasElements() const {
        if (!elementSetInitialized) const_cast<RectangularFilteredMeshBase<DIM>*>(this)->calculateElements();
    }

};

}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED_COMMON_H
