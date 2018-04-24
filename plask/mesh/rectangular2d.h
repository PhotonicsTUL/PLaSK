#ifndef PLASK__RECTANGULAR2D_H
#define PLASK__RECTANGULAR2D_H

/** @file
This file contains rectangular mesh for 2D space.
*/

#include <iterator>

#include "mesh.h"
#include "boundary.h"
#include "../utils/interpolation.h"
#include "../geometry/object.h"
#include "../geometry/space.h"
#include "../geometry/path.h"
#include "../math.h"
#include "../manager.h"

#include "axis1d.h"
#include "ordered1d.h"

namespace plask {

namespace details {

/**
 * Helper used by getLeftOfBoundary, etc.
 * @param[out] line index of point in @p axis which lies in bound [@p box_lower, @p box_upper] and is the nearest to @p box_lower,
 *  undefined if @c false was returned
 * @param[in] axis axis, 1D mesh
 * @param[in] box_lower, box_upper position of lower and upper box edges
 * @return @c true only if @p axis has point which lies in bounds [@p box_lower, @p box_upper]
 */
inline bool getLineLo(std::size_t& line, const MeshAxis& axis, double box_lower, double box_upper) {
    assert(box_lower <= box_upper);
    line = axis.findIndex(box_lower);
    return line != axis.size() && axis[line] <= box_upper;
}

/**
 * Helper used by getRightOfBoundary, etc.
 * @param[out] line index of point in @p axis which lies in bound [@p box_lower, @p box_upper] and is nearest to @p box_upper,
 *  undefined if @c false was returned
 * @param[in] axis axis, 1D mesh
 * @param[in] box_lower, box_upper position of lower and upper box edges
 * @return @c true only if @p axis has point which lies in bounds [@p box_lower, @p box_upper]
 */
inline bool getLineHi(std::size_t& line, const MeshAxis& axis, double box_lower, double box_upper) {
    assert(box_lower <= box_upper);
    line = axis.findIndex(box_upper);
    if (line != axis.size() && axis[line] == box_upper) return true;
    if (line == 0) return false;
    --line;
    return axis[line] >= box_lower;
}

/**
 * Helper used by getLeftOfBoundary, etc.
 * @param[out] begInd, endInd range [begInd, endInd) of indices in @p axis which show points which lie in bounds [@p box_lower, @p box_upper],
 *      undefined if @c false was returned
 * @param[in] axis axis, 1D mesh
 * @param[in] box_lower, box_upper position of lower and upper box edges
 * @return @c true only if some of @p axis points lies in bounds [@p box_lower, @p box_upper]
 */
inline bool getIndexesInBounds(std::size_t& begInd, std::size_t& endInd, const MeshAxis& axis, double box_lower, double box_upper) {
    if(box_lower > box_upper) return false;
    begInd = axis.findIndex(box_lower);
    endInd = axis.findIndex(box_upper);
    if (endInd != axis.size() && axis[endInd] == box_upper) ++endInd;    // endInd is exluded
    return begInd != endInd;
}

/**
 * Decrease @p index if @p real_pos is much closer to axis[index-1] than axis[index].
 * @param[in] axis axis of mesh
 * @param[in, out] index index such that axis[index] <= real_pos < axis[index+1], can be unchanged or decrement by one by this method
 * @param[in] real_pos position
 */
inline void tryMakeLower(const MeshAxis& axis, std::size_t& index, double real_pos) {
    if (index == 0) return;
    if ((real_pos - axis[index-1]) * 100.0 < (axis[index] - axis[index-1])) --index;
}

/**
 * Increase @p index if @p real_pos is much closer to axis[index] than axis[index-1].
 * @param[in] axis axis of mesh
 * @param[in, out] index index such that axis[index-1] <= real_pos < axis[index], can be unchanged or increment by one by this method
 * @param[in] real_pos position
 */
inline void tryMakeHigher(const MeshAxis& axis, std::size_t& index, double real_pos) {
    if (index == axis.size() || index == 0) return; //index == 0 means empty mesh
    if ((axis[index] - real_pos) * 100.0 < (axis[index] - axis[index-1])) ++index;
}

/**
 * Helper.
 * @param[out] begInd, endInd range [begInd, endInd) of indices in @p axis which show points which lie or almost lie in bounds [@p box_lower, @p box_upper],
 *      undefined if @c false was returned
 * @param[in] axis axis, 1D mesh
 * @param[in] box_lower, box_upper position of lower and upper box edges
 * @return @c true only if some of @p axis points (almost) lies in bounds [@p box_lower, @p box_upper]
 */
inline bool getIndexesInBoundsExt(std::size_t& begInd, std::size_t& endInd, const MeshAxis& axis, double box_lower, double box_upper) {
    getIndexesInBounds(begInd, endInd, axis, box_lower, box_upper);
    tryMakeLower(axis, begInd, box_lower);
    tryMakeHigher(axis, endInd, box_upper);
    return begInd != endInd;
}

/**
 * Get boundary which lies on chosen edge of boxes.
 * @param getBoxes functor which returns 0 or more boxes (vector of boxes)
 * @param getBoundaryForBox functor which returns boundary for box given as parameter, it chooses edge of box (for example this may call getLeftOfBoundary, etc.)
 * @return boundary which represents sum of boundaries returned by getBoundaryForBox for all boxes returned by getBoxes
 * @tparam MeshType RectangularMesh...
 */
template <typename MeshType, typename GetBoxes, typename GetBoundaryForBox>
inline typename MeshType::Boundary getBoundaryForBoxes(GetBoxes getBoxes, GetBoundaryForBox getBoundaryForBox) {
    return typename MeshType::Boundary(
        [=](const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) -> BoundaryWithMesh {
            std::vector<typename MeshType::Boundary> boundaries;
            std::vector<typename MeshType::Boundary::WithMesh> boundaries_with_meshes;
            auto boxes = getBoxes(geometry); // probably std::vector<BoxdirD>
            for (auto& box: boxes) {
                typename MeshType::Boundary boundary = getBoundaryForBox(box);
                typename MeshType::Boundary::WithMesh boundary_with_mesh = boundary(mesh, geometry);
                if (!boundary_with_mesh.empty()) {
                    boundaries.push_back(std::move(boundary));
                    boundaries_with_meshes.push_back(std::move(boundary_with_mesh));
                }
            }
            if (boundaries.empty()) return new EmptyBoundaryImpl();
            if (boundaries.size() == 1) return boundaries_with_meshes[0];
            return new SumBoundaryImpl<MeshType>(std::move(boundaries_with_meshes));
        }
    );
}

/*struct GetObjectBoundingBoxesCaller {

    shared_ptr<const GeometryObject> object;
    std::unique_ptr<PathHints> path;

    GetObjectBoundingBoxesCaller(shared_ptr<const GeometryObject> object, const PathHints* path)
        : object(object), path(path ? new PathHints(*path) : nullptr)
    {}

    auto operator()(const shared_ptr<const GeometryD<2>>& geometry) const -> decltype(geometry->getObjectBoundingBoxes(object))  {
        return geometry->getObjectBoundingBoxes(object, path.get());
    }

    auto operator()(const shared_ptr<const GeometryD<3>>& geometry) const -> decltype(geometry->getObjectBoundingBoxes(object))  {
        return geometry->getObjectBoundingBoxes(object, path.get());
    }

};*/

/**
 * Parse boundary from XML tag in format:
 * \<place side="i.e. left" [object="object name" [path="path name"] [geometry="name of geometry which is used by the solver"]]/>
 * @param boundary_desc XML reader which point to tag to read (after read it will be moved to end of this tag)
 * @param manager geometry manager
 * @param getXBoundary function which creates simple boundary, with edge of mesh, i.e. getLeftBoundary
 * @param getXOfBoundary function which creates simple boundary, with edge of object, i.e. getLeftOfBoundary
 * @return boundary which was read
 */
template <typename Boundary, int DIM>
inline Boundary parseBoundaryFromXML(XMLReader& boundary_desc, Manager& manager, Boundary (*getXBoundary)(),
                                     Boundary (*getXOfBoundary)(shared_ptr<const GeometryObject>, const PathHints*)) {
    plask::optional<std::string> of = boundary_desc.getAttribute("object");
    if (!of) {
        boundary_desc.requireTagEnd();
        return getXBoundary();
    } else {
        plask::optional<std::string> path_name = boundary_desc.getAttribute("path");
        boundary_desc.requireTagEnd();
        return getXOfBoundary(manager.requireGeometryObject(*of),
                              path_name ? &manager.requirePathHints(*path_name) : nullptr);
    }
}

}   // namespace details

/**
 * Rectilinear mesh in 2D space.
 *
 * Includes two 1D rectilinear meshes:
 * - axis0 (alternative names: tran(), ee_x(), rad_r())
 * - axis1 (alternative names: up(), ee_y(), rad_z())
 * Represent all points (x, y) such that x is in axis0 and y is in axis1.
 */
template<>
class PLASK_API RectangularMesh<2>: public MeshD<2> {

    typedef std::size_t index_ft(const RectangularMesh<2>* mesh, std::size_t axis0_index, std::size_t axis1_index);
    typedef std::size_t index01_ft(const RectangularMesh<2>* mesh, std::size_t mesh_index);

    // Our own virtual table, changeable in run-time:
    index_ft* index_f;
    index01_ft* index0_f;
    index01_ft* index1_f;

    const shared_ptr<MeshAxis>* minor_axis; ///< minor (changing fastest) axis
    const shared_ptr<MeshAxis>* major_axis; ///< major (changing slowest) axis

    void onAxisChanged(Event& e);

    void setChangeSignal(const shared_ptr<MeshAxis>& axis) { if (axis) axis->changedConnectMethod(this, &RectangularMesh<2>::onAxisChanged); }
    void unsetChangeSignal(const shared_ptr<MeshAxis>& axis) { if (axis) axis->changedDisconnectMethod(this, &RectangularMesh<2>::onAxisChanged); }

    void setAxis(const shared_ptr<MeshAxis>& axis, shared_ptr<MeshAxis> new_val);

  public:

    /**
     * Represent FEM-like element in RectangularMesh.
     */
    class PLASK_API Element {
        const RectangularMesh<2>& mesh;
        std::size_t index0, index1; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        public:

        /**
         * Construct element using mesh and element indexes.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param index0, index1 axis 0 and 1 indexes of element (equal to low corrner mesh indexes of element)
         */
        Element(const RectangularMesh<2>& mesh, std::size_t index0, std::size_t index1): mesh(mesh), index0(index0), index1(index1) {}

        /**
         * Construct element using mesh and element index.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param elementIndex index of element
         */
        Element(const RectangularMesh<2>& mesh, std::size_t elementIndex): mesh(mesh) {
            std::size_t v = mesh.getElementMeshLowIndex(elementIndex);
            index0 = mesh.index0(v);
            index1 = mesh.index1(v);
        }

        /// \return tran index of the element
        inline std::size_t getIndex0() const { return index0; }

        /// \return vert index of the element
        inline std::size_t getIndex1() const { return index1; }

        /// \return tran index of the left edge of the element
        inline std::size_t getLowerIndex0() const { return index0; }

        /// \return vert index of the bottom edge of the element
        inline std::size_t getLowerIndex1() const { return index1; }

        /// \return tran coordinate of the left edge of the element
        inline double getLower0() const { return mesh.axis[0]->at(index0); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower1() const { return mesh.axis[1]->at(index1); }

        /// \return tran index of the right edge of the element
        inline std::size_t getUpperIndex0() const { return index0+1; }

        /// \return vert index of the top edge of the element
        inline std::size_t getUpperIndex1() const { return index1+1; }

        /// \return tran coordinate of the right edge of the element
        inline double getUpper0() const { return mesh.axis[0]->at(getUpperIndex0()); }

        /// \return vert coordinate of the top edge of the element
        inline double getUpper1() const { return mesh.axis[1]->at(getUpperIndex1()); }

        /// \return size of the element in the tran direction
        inline double getSize0() const { return getUpper0() - getLower0(); }

        /// \return size of the element in the vert direction
        inline double getSize1() const { return getUpper1() - getLower1(); }

        /// \return vector indicating size of the element
        inline Vec<2, double> getSize() const { return getUpUp() - getLoLo(); }

        /// \return position of the middle of the element
        inline Vec<2, double> getMidpoint() const { return mesh.getElementMidpoint(index0, index1); }

        /// @return index of this element
        inline std::size_t getIndex() const { return mesh.getElementIndexFromLowIndexes(getLowerIndex0(), getLowerIndex1()); }

        /// \return this element as rectangular box
        inline Box2D toBox() const { return mesh.getElementBox(index0, index1); }

        /// \return total area of this element
        inline double getVolume() const { return getSize0() * getSize1(); }

        /// \return total area of this element
        inline double getArea() const { return getVolume(); }

        /// \return index of the lower left corner of this element
        inline std::size_t getLoLoIndex() const { return mesh.index(getLowerIndex0(), getLowerIndex1()); }

        /// \return index of the upper left corner of this element
        inline std::size_t getLoUpIndex() const { return mesh.index(getLowerIndex0(), getUpperIndex1()); }

        /// \return index of the lower right corner of this element
        inline std::size_t getUpLoIndex() const { return mesh.index(getUpperIndex0(), getLowerIndex1()); }

        /// \return index of the upper right corner of this element
        inline std::size_t getUpUpIndex() const { return mesh.index(getUpperIndex0(), getUpperIndex1()); }

        /// \return position of the lower left corner of this element
        inline Vec<2, double> getLoLo() const { return mesh(getLowerIndex0(), getLowerIndex1()); }

        /// \return position of the upper left corner of this element
        inline Vec<2, double> getLoUp() const { return mesh(getLowerIndex0(), getUpperIndex1()); }

        /// \return position of the lower right corner of this element
        inline Vec<2, double> getUpLo() const { return mesh(getUpperIndex0(), getLowerIndex1()); }

        /// \return position of the upper right corner of this element
        inline Vec<2, double> getUpUp() const { return mesh(getUpperIndex0(), getUpperIndex1()); }

    };

    /**
     * Wrapper to RectangularMesh which allow to access to FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    class PLASK_API Elements {

        static inline Element deref(const RectangularMesh<2>& mesh, std::size_t index) { return mesh.getElement(index); }
    public:
        typedef IndexedIterator<const RectangularMesh<2>, Element, deref> const_iterator;
        typedef const_iterator iterator;

        const RectangularMesh<2>* mesh;

        Elements(const RectangularMesh<2>* mesh): mesh(mesh) {}

        /**
         * Get @p i-th element.
         * @param i element index
         * @return @p i-th element
         */
        Element operator[](std::size_t i) const { return Element(*mesh, i); }

        /**
         * Get element with indices \p i0 and \p i1.
         * \param i0, i1 element index
         * \return element with indices \p i0 and \p i1
         */
        Element operator()(std::size_t i0, std::size_t i1) const { return Element(*mesh, i0, i1); }

        /**
         * Get number of elements.
         * @return number of elements
         */
        std::size_t size() const { return mesh->getElementsCount(); }

        /// @return iterator referring to the first element
        const_iterator begin() const { return const_iterator(mesh, 0); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(mesh, size()); }

    };

    /// Boundary type.
    typedef plask::Boundary<RectangularMesh<2>> Boundary;

    /// First and second coordinates of points in this mesh.
    const shared_ptr<MeshAxis> axis[2];

    /// Accessor to FEM-like elements.
    Elements elements() const { return Elements(this); }
    Elements getElements() const { return elements(); }

    Element element(std::size_t i0, std::size_t i1) const { return Element(*this, i0, i1); }
    Element getElement(std::size_t i0, std::size_t i1) const { return element(i0, i1); }

    /**
     * Get an element with a given index @p i.
     * @param i index of the element
     * @return the element
     */
    Element element(std::size_t i) const { return Element(*this, i); }

    /**
     * Get an element with a given index @p i.
     * @param i index of the element
     * @return the element
     */
    Element getElement(std::size_t i) const { return element(i); }

    /**
     * Iteration orders:
     * - 10 iteration order is:
     * (axis0[0], axis1[0]), (axis0[1], axis1[0]), ..., (axis0[axis0->size-1], axis1[0]), (axis0[0], axis1[1]), ..., (axis0[axis0->size()-1], axis1[axis[1]->size()-1])
     * - 01 iteration order is:
     * (axis0[0], axis1[0]), (axis0[0], axis1[1]), ..., (axis0[0], y[axis[1]->size-1]), (axis0[1], axis1[0]), ..., (axis0[axis0->size()-1], axis1[axis[1]->size()-1])
     * @see setIterationOrder, getIterationOrder, setOptimalIterationOrder
     */
    enum IterationOrder { ORDER_10, ORDER_01 };

    /**
     * Choose iteration order.
     * @param order iteration order to use
     * @see IterationOrder
     */
    void setIterationOrder(IterationOrder order);

    /**
     * Get iteration order.
     * @return iteration order used by this mesh
     * @see IterationOrder
     */
    IterationOrder getIterationOrder() const;

    /**
     * Set iteration order to the shortest axis changes fastest.
     */
    void setOptimalIterationOrder() {
        setIterationOrder(axis[0]->size() > axis[1]->size() ? ORDER_01 : ORDER_10);
    }

    /**
     * Construct mesh which has all axes of type OrderedAxis and all are empty.
     * @param iterationOrder iteration order
     */
    explicit RectangularMesh(IterationOrder iterationOrder = ORDER_01);

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh(shared_ptr<MeshAxis> axis0, shared_ptr<MeshAxis> axis1, IterationOrder iterationOrder = ORDER_01);

    /**
     * Copy constructor.
     * @param src mesh to copy
     * @param clone_axes whether axes of the @p src should be cloned (if true) or shared (if false; default)
     */
    RectangularMesh(const RectangularMesh<2>& src, bool clone_axes = false);

    ~RectangularMesh();

    const shared_ptr<MeshAxis> getAxis0() const { return axis[0]; }

    void setAxis0(shared_ptr<MeshAxis> a0) { setAxis(this->axis[0], a0); }

    const shared_ptr<MeshAxis> getAxis1() const { return axis[1]; }

    void setAxis1(shared_ptr<MeshAxis> a1) { setAxis(this->axis[1], a1); }



    /*
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate, or constructor argument for the first coordinate mesh
     * @param mesh1 mesh for the second coordinate, or constructor argument for the second coordinate mesh
     * @param iterationOrder iteration order
     */
    /*template <typename Mesh0CtorArg, typename Mesh1CtorArg>
    RectangularMesh(Mesh0CtorArg&& mesh0, Mesh1CtorArg&& mesh1, IterationOrder iterationOrder = ORDER_10):
        axis0(std::forward<Mesh0CtorArg>(mesh0)), axis1(std::forward<Mesh1CtorArg>(mesh1)) elements(this) {
        axis0->owner = this; axis[1]->owner = this;
        setIterationOrder(iterationOrder); }*/

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& tran() const { return axis[0]; }

    void setTran(shared_ptr<MeshAxis> a0) { setAxis0(a0); }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& vert() const { return axis[1]; }

    void setVert(shared_ptr<MeshAxis> a1) { setAxis1(a1); }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& ee_x() const { return axis[0]; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& ee_y() const { return axis[1]; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& rad_r() const { return axis[0]; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& rad_z() const { return axis[1]; }

    /**
     * Get numbered axis
     * @param n number of axis
     * @return n-th axis (cn)
     */
    const shared_ptr<MeshAxis>& getAxis(size_t n) const {
        if (n >= 2) throw Exception("Bad axis number");
        return axis[n];
    }

    /// \return major (changing slowest) axis
    inline const shared_ptr<MeshAxis> majorAxis() const {
        return *major_axis;
    }

    /// \return minor (changing fastest) axis
    inline const shared_ptr<MeshAxis> minorAxis() const {
        return *minor_axis;
    }

    /**
      * Compare meshes
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points regardless of iteration order
      */
    bool operator==(const RectangularMesh<2>& to_compare) const {
        return *axis[0] == *to_compare.axis[0] && *axis[1] == *to_compare.axis[1];
    }

    /**
     * Get number of points in mesh.
     * @return number of points in mesh
     */
    std::size_t size() const override { return axis[0]->size() * axis[1]->size(); }

    /**
     * Get maximum of sizes axis0 and axis1
     * @return maximum of sizes axis0 and axis1
     */
    std::size_t getMaxSize() const { return std::max(axis[0]->size(), axis[1]->size()); }

    /**
     * Get minimum of sizes axis0 and axis1
     * @return minimum of sizes axis0 and axis1
     */
    std::size_t getMinSize() const { return std::min(axis[0]->size(), axis[1]->size()); }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    void writeXML(XMLElement& object) const override;

    /// @return true only if there are no points in mesh
    bool empty() const override { return axis[0]->empty() || axis[1]->empty(); }

    /**
     * Calculate this mesh index using indexes of axis0 and axis1.
     * @param axis0_index index of axis0, from 0 to axis[0]->size()-1
     * @param axis1_index index of axis1, from 0 to axis[1]->size()-1
     * @return this mesh index, from 0 to size()-1
     */
    inline std::size_t index(std::size_t axis0_index, std::size_t axis1_index) const {
        return index_f(this, axis0_index, axis1_index);
    }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis[0]->size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
        return index0_f(this, mesh_index);
    }

    /**
     * Calculate index of y using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis1, from 0 to axis[1]->size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return index1_f(this, mesh_index);
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return mesh_index / (*minor_axis)->size();
    }

    /**
     * Calculate index of minor axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of minor axis, from 0 to minorAxis.size()-1
     */
    inline std::size_t minorIndex(std::size_t mesh_index) const {
        return mesh_index % (*minor_axis)->size();
    }

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @return point with given @p index
     */
    inline Vec<2, double> at(std::size_t index0, std::size_t index1) const {
        return Vec<2, double>(axis[0]->at(index0), axis[1]->at(index1));
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    virtual Vec<2, double> at(std::size_t index) const override {
        return Vec<2, double>(axis[0]->at(index0(index)), axis[1]->at(index1(index)));
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     * @see IterationOrder
     */
    inline Vec<2,double> operator[](std::size_t index) const {
        return Vec<2, double>(axis[0]->at(index0(index)), axis[1]->at(index1(index)));
    }

    /**
     * Get point with given x and y indexes.
     * @param axis0_index index of axis0, from 0 to axis0->size()-1
     * @param axis1_index index of axis1, from 0 to axis[1]->size()-1
     * @return point with given axis0 and axis1 indexes
     */
    inline Vec<2,double> operator()(std::size_t axis0_index, std::size_t axis1_index) const {
        return Vec<2, double>(axis[0]->at(axis0_index), axis[1]->at(axis1_index));
    }

    /**
     * Remove all points from mesh.
     */
    /*void clear() {
        axis0->clear();
        axis[1]->clear();
    }*/

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    shared_ptr<RectangularMesh> getMidpointsMesh();

    /**
     * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        auto p = flags.wrap(point);

        size_t index0_lo, index0_hi;
        double left, right;
        bool invert_left, invert_right;
        prepareInterpolationForAxis(*axis[0], flags, p.c0, 0, index0_lo, index0_hi, left, right, invert_left, invert_right);

        size_t index1_lo, index1_hi;
        double bottom, top;
        bool invert_bottom, invert_top;
        prepareInterpolationForAxis(*axis[1], flags, p.c1, 1, index1_lo, index1_hi, bottom, top, invert_bottom, invert_top);

        typename std::remove_const<typename std::remove_reference<decltype(data[0])>::type>::type
            data_lb = data[index(index0_lo, index1_lo)],
            data_rb = data[index(index0_hi, index1_lo)],
            data_rt = data[index(index0_hi, index1_hi)],
            data_lt = data[index(index0_lo, index1_hi)];
        if (invert_left)   { data_lb = flags.reflect(0, data_lb); data_lt = flags.reflect(0, data_lt); }
        if (invert_right)  { data_rb = flags.reflect(0, data_rb); data_rt = flags.reflect(0, data_rt); }
        if (invert_top)    { data_lt = flags.reflect(1, data_lt); data_rt = flags.reflect(1, data_rt); }
        if (invert_bottom) { data_lb = flags.reflect(1, data_lb); data_rb = flags.reflect(1, data_rb); }

        return flags.postprocess(point, interpolation::bilinear(left, right, bottom, top, data_lb, data_rb, data_rt, data_lt, p.c0, p.c1));
    }

    /**
     * Calculate (using nearest neighbor interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateNearestNeighbor(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type {
        auto p = flags.wrap(point);
        prepareNearestNeighborInterpolationForAxis(*axis[0], flags, p.c0, 0);
        prepareNearestNeighborInterpolationForAxis(*axis[1], flags, p.c1, 1);
        return flags.postprocess(point, data[this->index(axis[0]->findNearestIndex(p.c0), axis[1]->findNearestIndex(p.c1))]);
    }

    /**
     * Get number of elements (for FEM method) in the first direction.
     * @return number of elements in this mesh in the first direction (axis0 direction).
     */
    std::size_t getElementsCount0() const {
        const std::size_t s = axis[0]->size();
        return (s != 0)? s-1 : 0;
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in this mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        const std::size_t s = axis[1]->size();
        return (s != 0)? s-1 : 0;
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return (axis[0]->size() != 0 && axis[1]->size() != 0)?
            (axis[0]->size()-1) * (axis[1]->size()-1) : 0;
    }

    /**
     * Convert mesh index of bottom left element corner to index of this element.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        return mesh_index_of_el_bottom_left - mesh_index_of_el_bottom_left / (*minor_axis)->size();
    }

    /**
     * Convert mesh indexes of a bottom-left corner of an element to the index of this element.
     * @param axis0_index index of the corner along the axis0 (left), from 0 to axis[0]->size()-1
     * @param axis1_index index of the corner along the axis1 (bottom), from 0 to axis[1]->size()-1
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndexes(std::size_t axis0_index, std::size_t axis1_index) const {
        return getElementIndexFromLowIndex(index(axis0_index, axis1_index));
    }

    /**
     * Convert element index to mesh index of bottom-left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        return element_index + (element_index / ((*minor_axis)->size()-1));
    }

    /**
     * Convert an element index to mesh indexes of bottom-left corner of the element.
     * @param element_index index of the element, from 0 to getElementsCount()-1
     * @return axis 0 and axis 1 indexes of mesh,
     * you can easy calculate rest indexes of element corner by adding 1 to returned coordinates
     */
    Vec<2, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return Vec<2, std::size_t>(index0(bl_index), index1(bl_index));
    }

    /**
     * Get an area of a given element.
     * @param index0, index1 axis 0 and axis 1 indexes of the element
     * @return the area of the element with given indexes
     */
    double getElementArea(std::size_t index0, std::size_t index1) const {
        return (axis[0]->at(index0+1) - axis[0]->at(index0)) * (axis[1]->at(index1+1) - axis[1]->at(index1));
    }

    /**
     * Get an area of a given element.
     * @param element_index index of the element
     * @return the area of the element with given index
     */
    double getElementArea(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementArea(index0(bl_index), index1(bl_index));
    }

    /**
     * Get first coordinate of point in center of Elements.
     * @param index0 index of Elements (axis0 index)
     * @return first coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint0(std::size_t index0) const { return 0.5 * (axis[0]->at(index0) + axis[0]->at(index0+1)); }

    /**
     * Get second coordinate of point in center of Elements.
     * @param index1 index of Elements (axis1 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint1(std::size_t index1) const { return 0.5 * (axis[1]->at(index1) + axis[1]->at(index1+1)); }

    /**
     * Get point in center of Elements.
     * @param index0, index1 index of Elements
     * @return point in center of element with given index
     */
    Vec<2, double> getElementMidpoint(std::size_t index0, std::size_t index1) const {
        return vec(getElementMidpoint0(index0), getElementMidpoint1(index1));
    }

    /**
     * Get point in the center of an element.
     * @param element_index index of the element
     * @return point in the center of the element
     */
    Vec<2, double> getElementMidpoint(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementMidpoint(index0(bl_index), index1(bl_index));
    }

    /**
     * Get element as rectangle.
     * @param index0, index1 index of Elements
     * @return box of elements with given index
     */
    Box2D getElementBox(std::size_t index0, std::size_t index1) const {
        return Box2D(axis[0]->at(index0), axis[1]->at(index1), axis[0]->at(index0+1), axis[1]->at(index1+1));
    }

    /**
     * Get an element as a rectangle.
     * @param element_index index of the element
     * @return the element as a rectangle (box)
     */
    Box2D getElementBox(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementBox(index0(bl_index), index1(bl_index));
    }

  private:

    // Common code for: left, right, bottom, top boundries:
    struct BoundaryIteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        const RectangularMesh &mesh;

        std::size_t line;

        std::size_t index;

        BoundaryIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): mesh(mesh), line(line), index(index) {}

        virtual void increment() override { ++index; }

        virtual bool equal(const typename BoundaryLogicImpl::IteratorImpl& other) const override {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

    };

    // iterator over vertical line (from bottom to top). for left and right boundaries
    struct VerticalIteratorImpl: public BoundaryIteratorImpl {

        VerticalIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->line, this->index); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const override {
            return new VerticalIteratorImpl(*this);
        }
    };

    // iterator over horizonstal line (from left to right), for bottom and top boundaries
    struct HorizontalIteratorImpl: public BoundaryIteratorImpl {

        HorizontalIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->index, this->line); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const override {
            return new HorizontalIteratorImpl(*this);
        }
    };

    struct VerticalBoundary: public BoundaryWithMeshLogicImpl<RectangularMesh<2>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line;

        VerticalBoundary(const RectangularMesh<2>& mesh, std::size_t line_axis0): BoundaryWithMeshLogicImpl<RectangularMesh<2>>(mesh), line(line_axis0) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index0(mesh_index) == line;
        }

        Iterator begin() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, 0));
        }

        Iterator end() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, this->mesh.axis[1]->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis[1]->size();
        }
    };


    struct VerticalBoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<2>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line, beginInLineIndex, endInLineIndex;

        VerticalBoundaryInRange(const RectangularMesh<2>& mesh, std::size_t line_axis0, std::size_t beginInLineIndex, std::size_t endInLineIndex)
            : BoundaryWithMeshLogicImpl<RectangularMesh<2>>(mesh), line(line_axis0), beginInLineIndex(beginInLineIndex), endInLineIndex(endInLineIndex) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index0(mesh_index) == line && in_range(this->mesh.index1(mesh_index), beginInLineIndex, endInLineIndex);
        }

        Iterator begin() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, beginInLineIndex));
        }

        Iterator end() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, endInLineIndex));
        }

        std::size_t size() const override {
            return endInLineIndex - beginInLineIndex;
        }
    };

    struct HorizontalBoundary: public BoundaryWithMeshLogicImpl<RectangularMesh<2>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line;

        HorizontalBoundary(const RectangularMesh<2>& mesh, std::size_t line_axis1): BoundaryWithMeshLogicImpl<RectangularMesh<2>>(mesh), line(line_axis1) {}

        //virtual TopBoundary* clone() const { return new TopBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index1(mesh_index) == line;
        }

        Iterator begin() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, 0));
        }

        Iterator end() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, this->mesh.axis[0]->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis[0]->size();
        }
    };

    struct HorizontalBoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<2>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line, beginInLineIndex, endInLineIndex;

        HorizontalBoundaryInRange(const RectangularMesh<2>& mesh, std::size_t line_axis1, std::size_t beginInLineIndex, std::size_t endInLineIndex)
            : BoundaryWithMeshLogicImpl<RectangularMesh<2>>(mesh), line(line_axis1), beginInLineIndex(beginInLineIndex), endInLineIndex(endInLineIndex) {}
        //virtual TopBoundary* clone() const { return new TopBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index1(mesh_index) == line && in_range(this->mesh.index0(mesh_index), beginInLineIndex, endInLineIndex);
        }

        Iterator begin() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, beginInLineIndex));
        }

        Iterator end() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, endInLineIndex));
        }

        std::size_t size() const override {
            return endInLineIndex - beginInLineIndex;
        }
    };

    //TODO
    /*struct HorizontalLineBoundary: public BoundaryLogicImpl<RectangularMesh<2>> {

        double height;

        bool contains(const RectangularMesh &mesh, std::size_t mesh_index) const {
            return mesh.index1(mesh_index) == mesh.axis[1]->findNearestIndex(height);
        }
    };*/

public:
    // boundaries:

    template <typename Predicate>
    static Boundary getBoundary(Predicate predicate) {
        return Boundary(new PredicateBoundaryImpl<RectangularMesh<2>, Predicate>(predicate));
    }

    /**
     * Get boundary which show one vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @return boundary which show one vertical (from bottom to top) line in mesh
     */
    static Boundary getVerticalBoundaryAtLine(std::size_t line_nr_axis0) {
        return Boundary( [line_nr_axis0](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new VerticalBoundary(mesh, line_nr_axis0);
        } );
    }

    /**
     * Get boundary which show range in vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return boundary which show range in vertical (from bottom to top) line in mesh.
     */
    static Boundary getVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) {
        return Boundary( [=](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new VerticalBoundaryInRange(mesh, line_nr_axis0, indexBegin, indexEnd);
        } );
    }

    /**
     * Get boundary which show one vertical (from bottom to top) line in mesh which lies nearest given coordinate.
     * @param axis0_coord axis 0 coordinate
     * @return boundary which show one vertical (from bottom to top) line in mesh
     */
    static Boundary getVerticalBoundaryNear(double axis0_coord) {
        return Boundary( [axis0_coord](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new VerticalBoundary(mesh, mesh.axis[0]->findNearestIndex(axis0_coord));
        } );
    }

    /**
     * Get boundary which show one vertical (from bottom to top) segment in mesh which lies nearest given coordinate and has ends in given range
     * @param axis0_coord axis 0 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 1 coordinates
     * @return boundary which show one vertical (from bottom to top) segment in mesh
     */
    static Boundary getVerticalBoundaryNear(double axis0_coord, double from, double to) {
        return Boundary( [axis0_coord, from, to](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) -> BoundaryLogicImpl* {
            std::size_t begInd, endInd;
            if (!details::getIndexesInBoundsExt(begInd, endInd, *mesh.axis[1], from, to))
                return new EmptyBoundaryImpl();
            return new VerticalBoundaryInRange(mesh, mesh.axis[0]->findNearestIndex(axis0_coord), begInd, endInd);
        } );
    }

    /**
     * Get boundary which show one vertical, left (from bottom to top) line in mesh.
     * @return boundary which show left line in mesh
     */
    static Boundary getLeftBoundary() {
        return Boundary( [](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new VerticalBoundary(mesh, 0);
        } );
    }

    /**
     * Get boundary which show one vertical, right (from bottom to top) line in mesh.
     * @return boundary which show right line in mesh
     */
    static Boundary getRightBoundary() {
        return Boundary( [](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new VerticalBoundary(mesh, mesh.axis[0]->size()-1);
        } );
    }

    /**
     * Get boundary which lies on left edge of the @p box (at mesh line nearest left edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on left edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getLeftOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<2>>();
        return Boundary( [=](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (details::getLineLo(line, *mesh.axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd, endInd, *mesh.axis[1], box.lower.c1, box.upper.c1))
                return new VerticalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on right edge of the @p box (at mesh line nearest right edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on right edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getRightOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<2>>();
        return Boundary( [=](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (details::getLineHi(line, *mesh.axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd, endInd, *mesh.axis[1], box.lower.c1, box.upper.c1))
                return new VerticalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on bottom edge of the @p box (at mesh line nearest bottom edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on bottom edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getBottomOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<2>>();
        return Boundary( [=](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (details::getLineLo(line, *mesh.axis[1], box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd, endInd, *mesh.axis[0], box.lower.c0, box.upper.c0))
                return new HorizontalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on top edge of the @p box (at mesh line nearest top edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on top edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getTopOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<2>>();
        return Boundary( [=](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (details::getLineHi(line, *mesh.axis[1], box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd, endInd, *mesh.axis[0], box.lower.c0, box.upper.c0))
                return new HorizontalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on left edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of left edges of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2>::getLeftOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on left edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of left edges of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2>::getLeftOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on left edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of left edges of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path) {
        return path ? getLeftOfBoundary(object, *path) : getLeftOfBoundary(object);
    }

    /**
     * Get boundary which lies on right edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of right edges of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2>::getRightOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on right edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of right edges of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2>::getRightOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on right edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of right edges of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path) {
        return path ? getRightOfBoundary(object, *path) : getRightOfBoundary(object);
    }

    /**
     * Get boundary which lies on bottom edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of bottom edges of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2>::getBottomOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on bottom edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of bottom edges of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2>::getBottomOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on bottom edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of bottom edges of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path) {
        return path ? getBottomOfBoundary(object, *path) : getBottomOfBoundary(object);
    }

    /**
     * Get boundary which lies on top edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of top edges of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2>::getTopOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on top edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of top edges of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMesh<2> >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2>::getTopOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on top edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of top edges of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path) {
        return path ? getTopOfBoundary(object, *path) : getTopOfBoundary(object);
    }

    /**
     * Get boundary which show one horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @return boundary which show one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryAtLine(std::size_t line_nr_axis1) {
        return Boundary( [line_nr_axis1](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new HorizontalBoundary(mesh, line_nr_axis1);
        } );
    }

    /**
     * Get boundary which shows range in horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return boundary which show range in horizontal (from left to right) line in mesh.
     */
    static Boundary getHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) {
        return Boundary( [=](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new HorizontalBoundaryInRange(mesh, line_nr_axis1, indexBegin, indexEnd);
        } );
    }

    /**
     * Get boundary which shows one horizontal (from left to right) line in mesh which lies nearest given coordinate.
     * @param axis1_coord axis 1 coordinate
     * @return boundary which show one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryNear(double axis1_coord) {
        return Boundary( [axis1_coord](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new HorizontalBoundary(mesh, mesh.axis[1]->findNearestIndex(axis1_coord));
        } );
    }

    /**
     * Get boundary which show one horizontal (from left to right) segment in mesh which lies nearest given coordinate and has ends in given range.
     * @param axis1_coord axis 1 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 0 coordinates
     * @return boundary which show one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryNear(double axis1_coord, double from, double to) {
        return Boundary( [axis1_coord, from, to](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) -> BoundaryLogicImpl* {
            std::size_t begInd, endInd;
            if (!details::getIndexesInBoundsExt(begInd, endInd, *mesh.axis[0], from, to))
                return new EmptyBoundaryImpl();
            return new HorizontalBoundaryInRange(mesh, mesh.axis[1]->findNearestIndex(axis1_coord), begInd, endInd);
        } );
    }

    /**
     * Get boundary which shows one horizontal, top (from left to right) line in mesh.
     * @return boundary which show top line in mesh
     */
    static Boundary getTopBoundary() {
        return Boundary( [](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new HorizontalBoundary(mesh, mesh.axis[1]->size()-1);
        } );
    }

    /**
     * Get boundary which shows one horizontal, bottom (from left to right) line in mesh.
     * @return boundary which show bottom line in mesh
     */
    static Boundary getBottomBoundary() {
        return Boundary( [](const RectangularMesh<2>& mesh, const shared_ptr<const GeometryD<2>>&) {
            return new HorizontalBoundary(mesh, 0);
        } );
    }

    static Boundary getBoundary(const std::string& boundary_desc);

    static Boundary getBoundary(XMLReader& boundary_desc, Manager& manager);
};


template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<2>, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<2>>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis[0]->size() == 0 || src_mesh->axis[1]->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl< DstT, RectangularMesh<2>, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<2>, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<2>>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis[0]->size() == 0 || src_mesh->axis[1]->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl< DstT, RectangularMesh<2>, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

/**
 * Copy @p to_copy mesh using OrderedAxis to represent each axis in returned mesh.
 * @param to_copy mesh to copy
 * @return mesh with each axis of type OrderedAxis
 */
PLASK_API shared_ptr<RectangularMesh<2>> make_rectangular_mesh(const RectangularMesh<2>& to_copy);
inline shared_ptr<RectangularMesh<2>> make_rectangular_mesh(shared_ptr<const RectangularMesh<2>> to_copy) { return make_rectangular_mesh(*to_copy); }

template <>
inline Boundary<RectangularMesh<2>> parseBoundary<RectangularMesh<2>>(const std::string& boundary_desc, plask::Manager&) { return RectangularMesh<2>::getBoundary(boundary_desc); }

template <>
inline Boundary<RectangularMesh<2>> parseBoundary<RectangularMesh<2>>(XMLReader& boundary_desc, Manager& env) { return RectangularMesh<2>::getBoundary(boundary_desc, env); }

PLASK_API_EXTERN_TEMPLATE_CLASS(RectangularMesh<2>)

} // namespace plask

#include "rectangular_spline.h"

#endif // PLASK__RECTANGULAR2D_H
