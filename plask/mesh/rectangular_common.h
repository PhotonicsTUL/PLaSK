#ifndef PLASK__MESH__RECTANGULAR_COMMON_H
#define PLASK__MESH__RECTANGULAR_COMMON_H

#include "boundary.h"
#include "ordered1d.h"
#include "../geometry/path.h"
#include "../manager.h"

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
        [=](const MeshType& mesh, const shared_ptr<const GeometryD<MeshType::DIM>>& geometry) -> BoundaryNodeSet {
            std::vector<typename MeshType::Boundary> boundaries;
            std::vector<BoundaryNodeSet> boundaries_with_meshes;
            auto boxes = getBoxes(geometry); // probably std::vector<BoxdirD>
            for (auto& box: boxes) {
                typename MeshType::Boundary boundary = getBoundaryForBox(box);
                BoundaryNodeSet boundary_with_mesh = boundary(mesh, geometry);
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

struct RectangularMeshBase2D: public MeshD<2> {

    /// Boundary type.
    typedef plask::Boundary<RectangularMeshBase2D> Boundary;

    template <typename Predicate>
    static Boundary getBoundary(Predicate predicate) {
        return Boundary(new PredicateBoundaryImpl<RectangularMeshBase2D, Predicate>(predicate));
    }

    /**
     * Create a node set which includes one vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @return node set which includes one vertical (from bottom to top) line in mesh
     */
    virtual BoundaryNodeSet createVerticalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis0)) const {
        throw NotImplemented("createVerticalBoundaryAtLine(line_nr_axis0)");
    }

    /**
     * Get boundary which show one vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @return boundary which shows one vertical (from bottom to top) line in mesh
     */
    static Boundary getVerticalBoundaryAtLine(std::size_t line_nr_axis0) {
        return Boundary( [line_nr_axis0](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createVerticalBoundaryAtLine(line_nr_axis0);
        } );
    }

    /**
     * Create a node set which includes a range in vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return node set which includes range in vertical (from bottom to top) line in mesh.
     */
    virtual BoundaryNodeSet createVerticalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis0), std::size_t PLASK_UNUSED(indexBegin), std::size_t PLASK_UNUSED(indexEnd)) const {
        throw NotImplemented("createVerticalBoundaryAtLine(line_nr_axis0, indexBegin, indexEnd)");
    }

    /**
     * Get boundary which show range in vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return boundary which shows range in vertical (from bottom to top) line in mesh.
     */
    static Boundary getVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) {
        return Boundary( [=](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createVerticalBoundaryAtLine(line_nr_axis0, indexBegin, indexEnd);
        } );
    }

    /**
     * Create a node set which includes one vertical (from bottom to top) line in mesh which lies nearest given coordinate.
     * @param axis0_coord axis 0 coordinate
     * @return node set which includes one vertical (from bottom to top) line in mesh
     */
    virtual BoundaryNodeSet createVerticalBoundaryNear(double PLASK_UNUSED(axis0_coord)) const {
        throw NotImplemented("createVerticalBoundaryNear(axis0_coord)");
    }

    /**
     * Get boundary which show one vertical (from bottom to top) line in mesh which lies nearest given coordinate.
     * @param axis0_coord axis 0 coordinate
     * @return boundary which shows one vertical (from bottom to top) line in mesh
     */
    static Boundary getVerticalBoundaryNear(double axis0_coord) {
        return Boundary( [axis0_coord](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createVerticalBoundaryNear(axis0_coord);
        } );
    }

    /**
     * Create a node set which includes one vertical (from bottom to top) segment in mesh which lies nearest given coordinate and has ends in given range
     * @param axis0_coord axis 0 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 1 coordinates
     * @return node set which includes one vertical (from bottom to top) segment in mesh
     */
    virtual BoundaryNodeSet createVerticalBoundaryNear(double PLASK_UNUSED(axis0_coord), double PLASK_UNUSED(from), double PLASK_UNUSED(to)) const {
        throw NotImplemented("createVerticalBoundaryNear(axis0_coord, from, to)");
    }

    /**
     * Get boundary which show one vertical (from bottom to top) segment in mesh which lies nearest given coordinate and has ends in given range
     * @param axis0_coord axis 0 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 1 coordinates
     * @return boundary which shows one vertical (from bottom to top) segment in mesh
     */
    static Boundary getVerticalBoundaryNear(double axis0_coord, double from, double to) {
        return Boundary( [axis0_coord, from, to](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createVerticalBoundaryNear(axis0_coord, from, to);
        } );
    }

    /**
     * Create a node set which includes one vertical, left (from bottom to top) line in mesh.
     * @return node set which includes left line in mesh
     */
    virtual BoundaryNodeSet createLeftBoundary() const {
        throw NotImplemented("createLeftBoundary()");
    }

    /**
     * Get boundary which show one vertical, left (from bottom to top) line in mesh.
     * @return boundary which shows left line in mesh
     */
    static Boundary getLeftBoundary() {
        return Boundary( [](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createLeftBoundary();
        } );
    }

    /**
     * Create a node set which includes one vertical, right (from bottom to top) line in mesh.
     * @return node set which includes right line in mesh
     */
    virtual BoundaryNodeSet createRightBoundary() const {
        throw NotImplemented("createRightBoundary()");
    }


    /**
     * Get boundary which show one vertical, right (from bottom to top) line in mesh.
     * @return boundary which shows right line in mesh
     */
    static Boundary getRightBoundary() {
        return Boundary( [](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createRightBoundary();
        } );
    }

    /**
     * Create a node set which lies on left edge of the @p box (at mesh line nearest left edge and inside the box).
     * @param box box in which boundary should lie
     * @return node set which lies on left edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    virtual BoundaryNodeSet createLeftOfBoundary(const Box2D& PLASK_UNUSED(box)) const {
        throw NotImplemented("createLeftOfBoundary(box)");
    }

    /**
     * Get boundary which lies on left edge of the @p box (at mesh line nearest left edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on left edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getLeftOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMeshBase2D>();
        return Boundary( [=](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createLeftOfBoundary(box);
        } );
    }

    /**
     * Create a node set which lies on right edge of the @p box (at mesh line nearest right edge and inside the box).
     * @param box box in which boundary should lie
     * @return node set which lies on right edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    virtual BoundaryNodeSet createRightOfBoundary(const Box2D& PLASK_UNUSED(box)) const {
        throw NotImplemented("createRightOfBoundary(box)");
    }

    /**
     * Get boundary which lies on right edge of the @p box (at mesh line nearest right edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on right edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getRightOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMeshBase2D>();
        return Boundary( [=](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createRightOfBoundary(box);
        } );
    }

    /**
     * Create a node set which lies on bottom edge of the @p box (at mesh line nearest bottom edge and inside the box).
     * @param box box in which boundary should lie
     * @return node set which lies on bottom edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    virtual BoundaryNodeSet createBottomOfBoundary(const Box2D& PLASK_UNUSED(box)) const {
        throw NotImplemented("createBottomOfBoundary(box)");
    }

    /**
     * Get boundary which lies on bottom edge of the @p box (at mesh line nearest bottom edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on bottom edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getBottomOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMeshBase2D>();
        return Boundary( [=](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createBottomOfBoundary(box);
        } );
    }

    /**
     * Create a node set which lies on top edge of the @p box (at mesh line nearest top edge and inside the box).
     * @param box box in which boundary should lie
     * @return node set which lies on top edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    virtual BoundaryNodeSet createTopOfBoundary(const Box2D& PLASK_UNUSED(box)) const {
        throw NotImplemented("createTopOfBoundary(box)");
    }

    /**
     * Get boundary which lies on top edge of the @p box (at mesh line nearest top edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on top edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getTopOfBoundary(const Box2D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMeshBase2D>();
        return Boundary( [=](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createTopOfBoundary(box);
        } );
    }

    /**
     * Get boundary which lies on left edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of left edges of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMeshBase2D::getLeftOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on left edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of left edges of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMeshBase2D::getLeftOfBoundary(box); }
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
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMeshBase2D::getRightOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on right edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of right edges of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMeshBase2D::getRightOfBoundary(box); }
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
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMeshBase2D::getBottomOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on bottom edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of bottom edges of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMeshBase2D::getBottomOfBoundary(box); }
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
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMeshBase2D::getTopOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on top edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param object object included in @p geometry
     * @return boundary which represents sum of boundaries of top edges of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryObject> object) {
        return details::getBoundaryForBoxes< RectangularMeshBase2D >(
            [=](const shared_ptr<const GeometryD<2>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMeshBase2D::getTopOfBoundary(box); }
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
     * Create a node set which includes one horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @return node set which includes one horizontal (from left to right) line in mesh
     */
    virtual BoundaryNodeSet createHorizontalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis1)) const {
        throw NotImplemented("createHorizontalBoundaryAtLine(line_nr_axis1)");
    }

    /**
     * Get boundary which shows one horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @return boundary which shows one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryAtLine(std::size_t line_nr_axis1) {
        return Boundary( [line_nr_axis1](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createHorizontalBoundaryAtLine(line_nr_axis1);
        } );
    }

    /**
     * Create a node set which includes range in horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return node set which includes range in horizontal (from left to right) line in mesh.
     */
    virtual BoundaryNodeSet createHorizontalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis1), std::size_t PLASK_UNUSED(indexBegin), std::size_t PLASK_UNUSED(indexEnd)) const {
        throw NotImplemented("createHorizontalBoundaryAtLine(line_nr_axis1, indexBegin, indexEnd)");
    }

    /**
     * Get boundary which shows range in horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return boundary which shows range in horizontal (from left to right) line in mesh.
     */
    static Boundary getHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) {
        return Boundary( [=](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createHorizontalBoundaryAtLine(line_nr_axis1, indexBegin, indexEnd);
        } );
    }

    /**
     * Create a node set which includes one horizontal (from left to right) line in mesh which lies nearest given coordinate.
     * @param axis1_coord axis 1 coordinate
     * @return boundary which includes one horizontal (from left to right) line in mesh
     */
    virtual BoundaryNodeSet createHorizontalBoundaryNear(double PLASK_UNUSED(axis1_coord)) const {
        throw NotImplemented("createHorizontalBoundaryNear(axis1_coord)");
    }

    /**
     * Get boundary which shows one horizontal (from left to right) line in mesh which lies nearest given coordinate.
     * @param axis1_coord axis 1 coordinate
     * @return boundary which shows one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryNear(double axis1_coord) {
        return Boundary( [axis1_coord](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createHorizontalBoundaryNear(axis1_coord);
        } );
    }

    /**
     * Create a node set which includes one horizontal (from left to right) segment in mesh which lies nearest given coordinate and has ends in given range.
     * @param axis1_coord axis 1 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 0 coordinates
     * @return node set which includes one horizontal (from left to right) line in mesh
     */
    virtual BoundaryNodeSet createHorizontalBoundaryNear(double PLASK_UNUSED(axis1_coord), double PLASK_UNUSED(from), double PLASK_UNUSED(to)) const {
        throw NotImplemented("createHorizontalBoundaryNear(axis1_coord, from, to)");
    }

    /**
     * Get boundary which show one horizontal (from left to right) segment in mesh which lies nearest given coordinate and has ends in given range.
     * @param axis1_coord axis 1 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 0 coordinates
     * @return boundary which shows one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryNear(double axis1_coord, double from, double to) {
        return Boundary( [axis1_coord, from, to](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createHorizontalBoundaryNear(axis1_coord, from, to);
        } );
    }

    /**
     * Create node set which includes one horizontal, top (from left to right) line in mesh.
     * @return node set which includes top line in mesh
     */
    virtual BoundaryNodeSet createTopBoundary() const {
        throw NotImplemented("createTopBoundary()");
    }

    /**
     * Get boundary which shows one horizontal, top (from left to right) line in mesh.
     * @return boundary which show top line in mesh
     */
    static Boundary getTopBoundary() {
        return Boundary( [](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createTopBoundary();
        } );
    }

    /**
     * Create node set which includes one horizontal, bottom (from left to right) line in mesh.
     * @return node set which includes bottom line in mesh
     */
    virtual BoundaryNodeSet createBottomBoundary() const {
        throw NotImplemented("createBottomBoundary()");
    }

    /**
     * Get boundary which shows one horizontal, bottom (from left to right) line in mesh.
     * @return boundary which show bottom line in mesh
     */
    static Boundary getBottomBoundary() {
        return Boundary( [](const RectangularMeshBase2D& mesh, const shared_ptr<const GeometryD<2>>&) {
            return mesh.createBottomBoundary();
        } );
    }

    static Boundary getBoundary(const std::string &boundary_desc) {
        if (boundary_desc == "bottom") return getBottomBoundary();
        if (boundary_desc == "left") return getLeftBoundary();
        if (boundary_desc == "right") return getRightBoundary();
        if (boundary_desc == "top") return getTopBoundary();
        return Boundary();
    }

    static Boundary getBoundary(XMLReader &boundary_desc, Manager &manager) {
        auto side = boundary_desc.getAttribute("side");
        auto line = boundary_desc.getAttribute("line");
        if (side && line) {
            throw XMLConflictingAttributesException(boundary_desc, "side", "line");
        } else if (side) {
            if (*side == "bottom")
                return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getBottomBoundary, &getBottomOfBoundary);
            if (*side == "left")
                return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getLeftBoundary, &getLeftOfBoundary);
            if (*side == "right")
                return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getRightBoundary, &getRightOfBoundary);
            if (*side == "top")
                return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getTopBoundary, &getTopOfBoundary);
            throw XMLBadAttrException(boundary_desc, "side", *side);
        } else if (line) {
            double at = boundary_desc.requireAttribute<double>("at"),
                    start = boundary_desc.requireAttribute<double>("start"),
                    stop = boundary_desc.requireAttribute<double>("stop");
            boundary_desc.requireTagEnd();
            if (*line == "vertical")
                return getVerticalBoundaryNear(at, start, stop);
            if (*line == "horizontal")
                return getHorizontalBoundaryNear(at, start, stop);
            throw XMLBadAttrException(boundary_desc, "line", *line);
        }
        return Boundary();
    }

};

template <>
inline RectangularMeshBase2D::Boundary parseBoundary<RectangularMeshBase2D::Boundary>(const std::string& boundary_desc, plask::Manager&) { return RectangularMeshBase2D::getBoundary(boundary_desc); }

template <>
inline RectangularMeshBase2D::Boundary parseBoundary<RectangularMeshBase2D::Boundary>(XMLReader& boundary_desc, Manager& env) { return RectangularMeshBase2D::getBoundary(boundary_desc, env); }

}   // namespace plask

#endif // PLASK__MESH__RECTANGULAR_COMMON_H
