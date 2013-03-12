#ifndef PLASK__GENERATOR_RECTILINEAR_H
#define PLASK__GENERATOR_RECTILINEAR_H

#include "mesh.h"
#include "rectilinear.h"
#include <plask/geometry/path.h>

namespace plask {

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry
 * \return generated mesh
 */
shared_ptr<RectilinearMesh2D> makeGeometryGrid(const shared_ptr<GeometryObjectD<2>>& geometry);

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry
 * \return generated mesh
 */
shared_ptr<RectilinearMesh3D> makeGeometryGrid(const shared_ptr<GeometryObjectD<3>>& geometry);

/**
 * Generator of basic 2D geometry grid
 */
class RectilinearMesh2DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

    /// Should we add horizontal line at zero
    bool extend_to_zero;

  public:

    /**
     * Create generator
     * \param extend_to_zero indicates whether there always must be a line at tran = 0
     */
    RectilinearMesh2DSimpleGenerator(bool extend_to_zero=false): extend_to_zero(extend_to_zero) {}

    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryObjectD<2>>& geometry);
};


/**
 * Generator of basic 2D geometry grid
 */
struct RectilinearMesh3DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh3D> {

    virtual shared_ptr<RectilinearMesh3D> generate(const shared_ptr<GeometryObjectD<3>>& geometry);
};


class RectilinearMesh2DDivideGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

    size_t pre_divisions[2];
    size_t post_divisions[2];
    bool gradual;

    typedef std::map<std::pair<weak_ptr<const GeometryObjectD<2>>,PathHints>, std::set<double>> Refinements;

    Refinements refinements[2];

    RectilinearMesh1D get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<plask::GeometryObjectD<2>>& geometry, size_t dir);

  public:

    bool warn_multiple, ///< Warn if a single refinement points to more than one object.
         warn_missing,     ///< Warn if a defined refinement points to object absent from provided geometry.
         warn_outside;  ///< Warn if a defined refinement takes place outside of the pointed object.

    /**
     * Create new generator
     * \param prediv0 Initial mesh division in horizontal direction
     * \param postdiv0 Final mesh division in horizontal direction
     * \param prediv1 Initial mesh division in vertical direction (0 means the same as horizontal)
     * \param postdiv1 Final mesh division in vertical direction (0 means the same as horizontal)
    **/
    RectilinearMesh2DDivideGenerator(size_t prediv0=1, size_t postdiv0=1, size_t prediv1=0, size_t postdiv1=0) :
        gradual(true), warn_multiple(true), warn_missing(true), warn_outside(true)
    {
        pre_divisions[0] = prediv0;
        pre_divisions[1] = prediv1? prediv1 : prediv0;
        post_divisions[0] = postdiv0;
        post_divisions[1] = postdiv1? postdiv1 : postdiv0;
    }

    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryObjectD<2>>& geometry);

    /// Get initial division of the smallest object in the mesh
    inline std::pair<size_t,size_t> getPreDivision() const { return std::pair<size_t,size_t>(pre_divisions[0], pre_divisions[1]); }

    /// Set initial division of the smallest object in the mesh
    inline void setPreDivision(size_t div0, size_t div1=0) {
        pre_divisions[0] = div0;
        pre_divisions[1] = div1? div1 : div0;
        fireChanged();
    }

    /// Get final division of the smallest object in the mesh
    inline std::pair<size_t,size_t> getPostDivision() const { return std::pair<size_t,size_t>(post_divisions[0], post_divisions[1]); }

    /// Set final division of the smallest object in the mesh
    inline void setPostDivision(size_t div0, size_t div1=0) {
        post_divisions[0] = div0;
        post_divisions[1] = div1? div1 : div0;
        fireChanged();
    }

    /// \return true if the adjacent mesh elements cannot differ more than twice in size along each axis
    bool getGradual() const { return gradual; }

    /// \param value true if the adjacent mesh elements cannot differ more than twice in size along each axis
    void setGradual(bool value) {
        gradual = value;
        fireChanged();
    }


    /// \return map of refinements
    /// \param direction direction of the refinements
    const Refinements& getRefinements(Primitive<2>::Direction direction) const {
        return refinements[std::size_t(direction)];
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param object refined object
     * \param path additional path hints pointing to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(Primitive<2>::Direction direction, const weak_ptr<const GeometryObjectD<2>>& object, const PathHints& path, double position) {
        auto key = std::make_pair(object, path);
        refinements[std::size_t(direction)][key].insert(position);
        fireChanged();
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param object refined object
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(Primitive<2>::Direction direction, const weak_ptr<const GeometryObjectD<2>>& object, double position) {
        addRefinement(direction, object, PathHints(), position);
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param path path to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(Primitive<2>::Direction direction, const Path& path, double position) {
        addRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param subtree subtree to the refined object (only the last path is used)
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(Primitive<2>::Direction direction, const GeometryObject::Subtree& subtree, double position) {
        auto path = subtree.getLastPath();
        addRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove refinement to the mesh
     * \param direction direction in which the object was refined
     * \param object refined object
     * \param path additional path hints pointing to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(Primitive<2>::Direction direction, const weak_ptr<const GeometryObjectD<2>>& object, const PathHints& path, double position) {
        auto key = std::make_pair(object, path);
        auto ref = refinements[std::size_t(direction)].find(key);
        if (ref == refinements[std::size_t(direction)].end()) throw BadInput("RectilinearMesh2DDivideGenerator", "There are no refinements for specified geometry object.");
        auto oposition = ref->second.find(position);
        if (oposition == ref->second.end()) throw BadInput("RectilinearMesh2DDivideGenerator", "Specified geometry object does not have refinements at %1%.", *oposition);
        ref->second.erase(oposition);
        if (ref->second.empty()) refinements[std::size_t(direction)].erase(ref);
        fireChanged();
    }

    /**
     * Remove refinement to the mesh
     * \param direction direction in which the object was refined
     * \param object refined object
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(Primitive<2>::Direction direction, const weak_ptr<const GeometryObjectD<2>>& object, double position) {
        removeRefinement(direction, object, PathHints(), position);
    }

    /**
     * Remove refinement from the mesh
     * \param direction direction in which the object was refined
     * \param path path to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(Primitive<2>::Direction direction, const Path& path, double position) {
        removeRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove refinement from the mesh
     * \param direction direction in which the object was refined
     * \param subtree subtree to the refined object (only the last path is used)
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(Primitive<2>::Direction direction, const GeometryObject::Subtree& subtree, double position) {
        auto path = subtree.getLastPath();
        removeRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove all refinements from the object
     * \param object refined object
     * \param path additional path hints pointing to the refined object
     */
    void removeRefinements(const weak_ptr<const GeometryObjectD<2>>& object, const PathHints& path=PathHints()) {
        auto key = std::make_pair(object, path);
        auto ref0 = refinements[0].find(key);
        auto ref1 = refinements[1].find(key);
        if (ref0 == refinements[0].end() && ref1 == refinements[1].end())
            throw BadInput("RectilinearMesh2DDivideGenerator", "There are no refinements for specified geometry object.");
        else {
            if (ref0 != refinements[0].end()) refinements[0].erase(ref0);
            if (ref1 != refinements[1].end()) refinements[1].erase(ref1);
            fireChanged();
        }
    }

    /**
     * Remove all refinements from all objects
     */
    void clearRefinements() {
        refinements[0].clear();
        refinements[1].clear();
        fireChanged();
    }

    /**
     * Remove all refinements from the object
     * \param path path to the refined object
     */
    void removeRefinements(const Path& path) {
        removeRefinements(dynamic_pointer_cast<const GeometryObjectD<2>>(path.back()), PathHints(path));
    }

    /**
     * Remove all refinements from the object
     * \param subtree subtree to the refined object (only the last path is used)
     */
    void removeRefinements(const GeometryObject::Subtree& subtree) {
        auto path = subtree.getLastPath();
        removeRefinements(dynamic_pointer_cast<const GeometryObjectD<2>>(path.back()), PathHints(path));
    }

};

} // namespace plask

#endif // PLASK__GENERATOR_RECTILINEAR_H
