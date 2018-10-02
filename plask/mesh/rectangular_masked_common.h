#ifndef PLASK__RECTANGULAR_MASKED_COMMON_H
#define PLASK__RECTANGULAR_MASKED_COMMON_H

#include <functional>

#include "rectangular.h"
#include "../utils/numbers_set.h"

#include <boost/thread.hpp>


namespace plask {

/**
 * Common base class for RectangularMaskedMesh 2D and 3D.
 *
 * Do not use directly.
 */
template <int DIM>
struct RectangularMaskedMeshBase: public RectangularMeshBase<DIM> {

    /// Maximum distance from boundary to include in the inerpolation
    constexpr static double MIN_DISTANCE = 1e-9; // 1 femtometer

    /// Full, rectangular, wrapped mesh.
    RectangularMesh<DIM> fullMesh;

    using typename MeshD<DIM>::LocalCoords;

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
    template <typename MaskedMeshType>
    struct ElementsBase {

        using Element = typename MaskedMeshType::Element;

        /**
         * Iterator over elements.
         *
         * One can use:
         * - getIndex() method of the iterator to get index of the element,
         * - getNumber() method of the iterator to get index of the element in the wrapped mesh.
         */
        class const_iterator: public Set::ConstIteratorFacade<const_iterator, Element> {

            const MaskedMeshType* maskedMesh;

            Element dereference() const {
                return Element(*maskedMesh, this->getIndex(), this->getNumber());
            }

            friend class boost::iterator_core_access;

        public:

            template <typename... CtorArgs>
            explicit const_iterator(const MaskedMeshType& maskedMesh, CtorArgs&&... ctorArgs)
                : Set::ConstIteratorFacade<const_iterator, Element>(std::forward<CtorArgs>(ctorArgs)...), maskedMesh(&maskedMesh) {}

            const Set& set() const { return maskedMesh->elementSet; }
        };

        /// Iterator over elments. The same as const_iterator, since non-const iterators are not supported.
        typedef const_iterator iterator;

        const MaskedMeshType* maskedMesh;

        explicit ElementsBase(const MaskedMeshType& maskedMesh): maskedMesh(&maskedMesh) {}

        /**
         * Get number of elements.
         * @return number of elements
         */
        std::size_t size() const { return maskedMesh->getElementsCount(); }

        /// @return iterator referring to the first element
        const_iterator begin() const { return const_iterator(*maskedMesh, 0, maskedMesh->elementSet.segments.begin()); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(*maskedMesh, size(), maskedMesh->elementSet.segments.end()); }

        /**
         * Get @p i-th element.
         * @param i element index
         * @return @p i-th element
         */
        Element operator[](std::size_t i) const { return Element(*maskedMesh, i); }

    };  // struct Elements

    /**
     * Base class for element meshes with common code for 2D and 3D.
     */
    template <typename MaskedMeshType>
    struct ElementMeshBase: MeshD<DIM> {

        using Element = typename MaskedMeshType::Element;

        /// Iterator over elements.
        class const_iterator: public Set::ConstIteratorFacade<const_iterator, LocalCoords> {

            const MaskedMeshType* originalMesh;

            LocalCoords dereference() const {
                return Element(*originalMesh, this->getIndex(), this->getNumber()).getMidpoint();
            }

            friend class boost::iterator_core_access;

        public:

            template <typename... CtorArgs>
            explicit const_iterator(const MaskedMeshType& originalMesh, CtorArgs&&... ctorArgs)
                : Set::ConstIteratorFacade<const_iterator, LocalCoords>(std::forward<CtorArgs>(ctorArgs)...), originalMesh(&originalMesh) {}

            const Set& set() const { return originalMesh->elementSet; }
        };

        /// Iterator over elments. The same as const_iterator, since non-const iterators are not supported.
        typedef const_iterator iterator;

        /// @return iterator referring to the first element
        const_iterator begin() const { return const_iterator(*originalMesh, 0, originalMesh->elementSet.segments.begin()); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(*originalMesh, size(), originalMesh->elementSet.segments.end()); }

        const MaskedMeshType* originalMesh;
        RectangularMesh<DIM> fullMesh;

        explicit ElementMeshBase(const MaskedMeshType* originalMesh):
            originalMesh(originalMesh), fullMesh(*originalMesh->fullMesh.getElementMesh()) {}

        explicit ElementMeshBase(const MaskedMeshType& originalMesh): ElementMeshBase(&originalMesh) {}

        /**
         * Get number of elements.
         * @return number of elements
         */
        std::size_t size() const override { return originalMesh->getElementsCount(); }

        LocalCoords at(std::size_t index) const override {
            return fullMesh.at(originalMesh->elementSet.at(index));
        }

        bool empty() const override { return originalMesh->elementSet.empty(); }

    };  // ElementMeshBase

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

    /// Returned by some methods to signalize that element or node (with given index(es)) is not included in the mesh.
    enum:std::size_t { NOT_INCLUDED = Set::NOT_INCLUDED };

    /// Construct an empty mesh. One should use reset() method before using it.
    RectangularMaskedMeshBase() = default;

    /// Constructor which allows us to construct midpoints mesh.
    RectangularMaskedMeshBase(const RectangularMesh<DIM>& rectangularMesh, Set nodeSet, bool clone_axes = false)
        : fullMesh(rectangularMesh, clone_axes), nodeSet(std::move(nodeSet)), elementSetInitialized(false)
    {
        resetBoundyIndex();
        // TODO faster algorithm could iterate over segments
        for (auto nodeIndex: nodeSet) {
            auto indexes = rectangularMesh.indexes(nodeIndex);
            for (int d = 0; d < DIM; ++d) {
                if (indexes[d] < boundaryIndex[d].lo) boundaryIndex[d].lo = indexes[d];
                if (indexes[d] > boundaryIndex[d].up) boundaryIndex[d].up = indexes[d];
            }
        }
    }

    /**
     * Construct a mesh by wrap of a given @p rectangularMesh.
     * @param rectangularMesh mesh to wrap (it is copied by the constructor)
     * @param select_all whether select all nodes (if true) or do not select any nodes (if false; default) of @p rectangularMesh
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if true) or shared (if false; default)
     */
    RectangularMaskedMeshBase(const RectangularMesh<DIM>& rectangularMesh, bool select_all = false, bool clone_axes = false)
        : fullMesh(rectangularMesh, clone_axes)
    {
        if (select_all) {
            selectAll();
        } else
            resetBoundyIndex();
    }

    /**
     * Iterator over nodes coordinates. It implements const_iterator for masked meshes.
     *
     * Iterator of this type is faster than IndexedIterator used by parent class of masked meshes,
     * as it has constant time dereference operation while <code>at</code> method has logarithmic time complexity.
     *
     * One can use:
     * - getIndex() method of the iterator to get index of the node,
     * - getNumber() method of the iterator to get index of the node in the wrapped mesh.
     */
    class const_iterator: public CompressedSetOfNumbers<std::size_t>::ConstIteratorFacade<const_iterator, LocalCoords> {

        friend class boost::iterator_core_access;

        const RectangularMaskedMeshBase* mesh;

        LocalCoords dereference() const {
            return mesh->fullMesh.at(this->getNumber());
        }

    public:

        template <typename... CtorArgs>
        explicit const_iterator(const RectangularMaskedMeshBase& mesh, CtorArgs&&... ctorArgs)
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
     * Select all elements of wrapped mesh.
     */
    void selectAll() {
        this->nodeSet.assignRange(fullMesh.size());
        this->elementSet.assignRange(fullMesh.getElementsCount());
        for (int d = 0; d < DIM; ++d) {
            boundaryIndex[d].lo = 0;
            boundaryIndex[d].up = fullMesh.axis[d]->size()-1;
        }
    }

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
        return fullMesh.getElementsCount0();
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in the full rectangular mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        return fullMesh.getElementsCount1();
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return ensureHasElements().size();
    }

    /**
     * Convert mesh index of bottom left element corner to index of this element.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        return ensureHasElements().indexOf(fullMesh.getElementIndexFromLowIndex(nodeSet.at(mesh_index_of_el_bottom_left)));
    }

    /**
     * Convert element index to mesh index of bottom-left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        return nodeSet.indexOf(fullMesh.getElementMeshLowIndex(ensureHasElements().at(element_index)));
    }

    /**
     * Convert an element index to mesh indexes of bottom-left corner of the element.
     * @param element_index index of the element, from 0 to getElementsCount()-1
     * @return axis 0 and axis 1 indexes of mesh,
     * you can easy calculate rest indexes of element corner by adding 1 to returned coordinates
     */
    Vec<DIM, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        return fullMesh.getElementMeshLowIndexes(ensureHasElements().at(element_index));
    }

    /**
     * Get an area of a given element.
     * @param element_index index of the element
     * @return the area of the element with given index
     */
    double getElementArea(std::size_t element_index) const {
        return fullMesh.getElementArea(ensureHasElements().at(element_index));
    }

    /**
     * Get first coordinate of point in the center of an elements.
     * @param index0 index of the element (axis0 index)
     * @return first coordinate of the point in the center of the element
     */
    double getElementMidpoint0(std::size_t index0) const {
        return fullMesh.getElementMidpoint0(index0);
    }

    /**
     * Get second coordinate of point in the center of an elements.
     * @param index1 index of the element (axis1 index)
     * @return second coordinate of the point in the center of the element
     */
    double getElementMidpoint1(std::size_t index1) const {
        return fullMesh.getElementMidpoint1(index1);
    }

    /**
     * Get point in the center of an element.
     * @param element_index index of Elements
     * @return point in center of element with given index
     */
    Vec<DIM, double> getElementMidpoint(std::size_t element_index) const {
        return fullMesh.getElementMidpoint(ensureHasElements().at(element_index));
    }

    /**
     * Get an element as a rectangle.
     * @param element_index index of the element
     * @return the element as a rectangle (box)
     */
    typename Primitive<DIM>::Box getElementBox(std::size_t element_index) const {
        return fullMesh.getElementBox(ensureHasElements().at(element_index));
    }

  protected:    // constructing elementSet from nodes set (element is chosen when all its vertices are chosen) on-deamand

    /// Only one thread can calculate elementSet
    DontCopyThisField<boost::mutex> writeElementSet;

    /// Whether elementSet is initialized (default for most contructors)
    bool elementSetInitialized = true;

  private:

    /*bool restVerticesIncluded(const RectangularMesh2D::Element& el) const {
        return //nodeSet.includes(el.getLoLoIndex()) &&
               nodeSet.includes(el.getUpLoIndex()) &&
               nodeSet.includes(el.getLoUpIndex()) &&
               nodeSet.includes(el.getUpUpIndex());
    }
    bool restVerticesIncluded(const RectangularMesh3D::Element& el) const {
        return //nodeSet.includes(el.getLoLoLoIndex()) &&
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
        for (auto lowNode: nodeSet) {
            if (!fullMesh.isLowIndexOfElement(lowNode)) continue;
            auto elementIndex = fullMesh.getElementIndexFromLowIndex(lowNode);
            if (restVerticesIncluded(fullMesh.getElement(elementIndex))) elementSet.push_back(elementIndex);
        }
        elementSetInitialized = true;
    }*/ // ^ generic algorithm, slow but probably correct

    template <int d = DIM>
    typename std::enable_if<d == 2>::type calculateElements() {
        boost::lock_guard<boost::mutex> lock((boost::mutex&)writeElementSet);
        if (elementSetInitialized) return;  // another thread has initilized elementSet just when we waited for mutex

        if (fullMesh.axis[0]->size() <= 1 || fullMesh.axis[1]->size() <= 1) {
            elementSetInitialized = true;
            return;
        }

        elementSet = nodeSet.transformed([] (std::size_t&, std::size_t& e) { --e; });   // same as nodeSet.intersected(nodeSet.shiftedLeft(1))
        auto minor_axis_size = fullMesh.minorAxis()->size();
        elementSet = elementSet.intersection(elementSet.shiftedLeft(minor_axis_size));
        // now elementSet includes all low indexes which have other corners of elements (plus some indexes in the last column)
        // we have to transform low indexes to indexes of elements:
        elementSet = elementSet.transformed([&, minor_axis_size] (std::size_t& b, std::size_t& e) {  // here: 0 <= b < e
            if (e % minor_axis_size == 0) --e;  // end of segment cannot lie at the last column, as getElementIndexFromLowIndex confuses last column with the first element in the next row
            // no need to fix b (if b % minor_axis_size == 0) as rounding in getElementIndexFromLowIndex will give good value (as getElementIndexFromLowIndex(b)==getElementIndexFromLowIndex(b+1))
            b = fullMesh.getElementIndexFromLowIndex(b);
            e = fullMesh.getElementIndexFromLowIndex(e);
        });

        elementSetInitialized = true;
    }

    template <int d = DIM>
    typename std::enable_if<d == 3>::type calculateElements() {
        boost::lock_guard<boost::mutex> lock((boost::mutex&)writeElementSet);
        if (elementSetInitialized) return;  // another thread has initilized elementSet just when we waited for mutex

        if (fullMesh.axis[0]->size() <= 1 || fullMesh.axis[1]->size() <= 1 || fullMesh.axis[2]->size() <= 1) {
            elementSetInitialized = true;
            return;
        }

        elementSet = nodeSet.transformed([] (std::size_t&, std::size_t& e) { --e; });   // same as nodeSet.intersected(nodeSet.shiftedLeft(1))
        auto minor_axis_size = fullMesh.minorAxis()->size();
        elementSet = elementSet.intersection(elementSet.shiftedLeft(minor_axis_size));
        auto medium_axis_size = fullMesh.mediumAxis()->size();
        elementSet = elementSet.intersection(elementSet.shiftedLeft(minor_axis_size*medium_axis_size));
        // now elementSet includes all low indexes which have other corners of elements (plus some indexes in the last column)
        // we have to transform low indexes to indexes of elements:
        elementSet = elementSet.transformed([&, minor_axis_size, medium_axis_size] (std::size_t& b, std::size_t& e) {  // here: 0 <= b < e
            const std::size_t b_div = b / minor_axis_size;
            if (b_div % medium_axis_size == (medium_axis_size-1))
                b = (b_div+1) * minor_axis_size;    // first index in the next plane
            b = fullMesh.getElementIndexFromLowIndex(b);

            const std::size_t e_div = (e-1) / minor_axis_size;    // e-1 is the last index of the range
            if (e_div % medium_axis_size == (medium_axis_size-1))
                e = e_div * minor_axis_size - 1;    // -1 to not be divisible by minor_axis_size
            else if (e % minor_axis_size == 0) --e;
            e = fullMesh.getElementIndexFromLowIndex(e);
        });

        elementSetInitialized = true;
    }

  protected:
    /// Ensure that elementSet is calculated (calculate it if it is not)
    const Set& ensureHasElements() const {
        if (!elementSetInitialized) const_cast<RectangularMaskedMeshBase<DIM>*>(this)->calculateElements();
        return elementSet;
    }

};


}   // namespace plask

#endif // PLASK__RECTANGULAR_MASKED_COMMON_H
