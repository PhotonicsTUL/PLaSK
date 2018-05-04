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

}   // namespace plask

#endif // PLASK__MESH__RECTANGULAR_COMMON_H
