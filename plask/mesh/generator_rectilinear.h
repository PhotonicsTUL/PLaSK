#ifndef PLASK__GENERATOR_RECTILINEAR_H
#define PLASK__GENERATOR_RECTILINEAR_H

#include "mesh.h"
#include "mesh1d.h"
#include "rectilinear.h"
#include <plask/geometry/path.h>

namespace plask {

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry
 * \return generated mesh
 */
shared_ptr<RectilinearMesh2D> makeGeometryGrid(const shared_ptr<GeometryObjectD<2>>& geometry, double min_ply = 0.01, long max_points = 10);

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry
 * \return generated mesh
 */
shared_ptr<RectilinearMesh3D> makeGeometryGrid(const shared_ptr<GeometryObjectD<3>>& geometry, double min_ply = 0.01, long max_points = 10);

/**
 * Generator of basic 2D geometry grid
 */
class RectilinearMesh1DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh1D> {

    /// Minimum ply after split single, non-solid layer.
    double min_ply;

    /// Maximum points to split single, non-solid layer (more important than min_ply).
    long max_points;

    /// Should we add line at horizontal zero
    bool extend_to_zero;

    shared_ptr<RectilinearMesh1D> makeGeometryGrid1D(const shared_ptr<GeometryObjectD<2>>& geometry);

  public:

    /**
     * Create generator
     * \param extend_to_zero indicates whether there always must be a line at tran = 0
     * \param min_ply minimum ply after split single, non-solid layer
     * \param max_points maximum points to split single, non-solid layer (more important than min_ply)
     */
    RectilinearMesh1DSimpleGenerator(bool extend_to_zero=false, double min_ply = 0.01, long max_points = 10)
        : min_ply(min_ply), max_points(max_points), extend_to_zero(extend_to_zero) {}

    virtual shared_ptr<RectilinearMesh1D> generate(const shared_ptr<GeometryObjectD<2>>& geometry);
};


/**
 * Generator of basic 2D geometry grid
 */
class RectilinearMesh2DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

    /// Minimum ply after split single, non-solid layer.
    double min_ply;

    /// Maximum points to split single, non-solid layer (more important than min_ply).
    long max_points;

    /// Should we add line at horizontal zero
    bool extend_to_zero;

  public:

    /**
     * Create generator
     * \param extend_to_zero indicates whether there always must be a line at tran = 0
     * \param min_ply minimum ply after split single, non-solid layer
     * \param max_points maximum points to split single, non-solid layer (more important than min_ply)
     */
    RectilinearMesh2DSimpleGenerator(bool extend_to_zero=false, double min_ply = 0.01, long max_points = 10)
        : min_ply(min_ply), max_points(max_points), extend_to_zero(extend_to_zero) {}

    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryObjectD<2>>& geometry);
};

/**
 * Generator of 2D geometry grid using other generator for horizontal axis
 */
class RectilinearMesh2DFrom1DGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

    shared_ptr<MeshGeneratorOf<RectilinearMesh1D>> horizontal_generator;

  public:

    /**
     * Create generator
     * \param extend_to_zero indicates whether there always must be a line at tran = 0
     */
    RectilinearMesh2DFrom1DGenerator(const shared_ptr<MeshGeneratorOf<RectilinearMesh1D>>& source):
        horizontal_generator(source) {}

    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryObjectD<2>>& geometry);
};



/**
 * Generator of basic 3D geometry grid
 */
struct RectilinearMesh3DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh3D> {

    /// Minimum ply after split single, non-solid layer.
    double min_ply;

    /// Maximum points to split single, non-solid layer (more important than min_ply).
    long max_points;

public:

    /**
     * Create generator
     * \param extend_to_zero indicates whether there always must be a line at tran = 0
     * \param min_ply minimum ply after split single, non-solid layer
     * \param max_points maximum points to split single, non-solid layer (more important than min_ply)
     */
    RectilinearMesh3DSimpleGenerator(double min_ply = 0.01, long max_points = 10)
        : min_ply(min_ply), max_points(max_points) {}

    virtual shared_ptr<RectilinearMesh3D> generate(const shared_ptr<GeometryObjectD<3>>& geometry);
};

/**
 * Dividing generator ensuring no rapid change of element size
 */
template <int dim>
struct RectilinearMeshDivideGenerator: public MeshGeneratorOf<RectangularMesh<dim,RectilinearAxis>> {

    size_t pre_divisions[dim];
    size_t post_divisions[dim];
    bool gradual;

    using MeshGeneratorOf<RectangularMesh<dim,RectilinearAxis>>::DIM;

    typedef std::map<std::pair<weak_ptr<const GeometryObjectD<DIM>>,PathHints>, std::set<double>> Refinements;

    Refinements refinements[dim];

    RectilinearAxis getAxis(const RectilinearAxis& initial, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir);

    bool warn_multiple, ///< Warn if a single refinement points to more than one object.
         warn_missing,     ///< Warn if a defined refinement points to object absent from provided geometry.
         warn_outside;  ///< Warn if a defined refinement takes place outside of the pointed object.

    /**
     * Create new generator
    **/
    RectilinearMeshDivideGenerator() :
        gradual(true), warn_multiple(true), warn_missing(true), warn_outside(true)
    {
        for (int i = 0; i != dim; ++i) {
            pre_divisions[i] = 1;
            post_divisions[i] = 1;
        }
    }

    shared_ptr<RectangularMesh<dim,RectilinearAxis>>
    generate(const shared_ptr<GeometryObjectD<DIM>>& geometry);

    /// Get initial division of the smallest object in the mesh
    inline size_t getPreDivision(typename Primitive<DIM>::Direction direction) const {
        assert(size_t(direction) <= dim);
        return pre_divisions[size_t(direction)];
    }

    /// Set initial division of the smallest object in the mesh
    inline void setPreDivision(typename Primitive<DIM>::Direction direction, size_t div) {
        assert(size_t(direction) <= dim);
        pre_divisions[size_t(direction)] = div;
        this->fireChanged();
    }

    /// Get final division of the smallest object in the mesh
    inline size_t getPostDivision(typename Primitive<DIM>::Direction direction) const {
        assert(size_t(direction) <= dim);
        return post_divisions[size_t(direction)];
    }

    /// Set final division of the smallest object in the mesh
    inline void setPostDivision(typename Primitive<DIM>::Direction direction, size_t div) {
        assert(size_t(direction) <= dim);
        post_divisions[size_t(direction)] = div;
        this->fireChanged();
    }

    /// \return true if the adjacent mesh elements cannot differ more than twice in size along each axis
    bool getGradual() const { return gradual; }

    /// \param value true if the adjacent mesh elements cannot differ more than twice in size along each axis
    void setGradual(bool value) {
        gradual = value;
        this->fireChanged();
    }


    /// \return map of refinements
    /// \param direction direction of the refinements
    const Refinements& getRefinements(typename Primitive<DIM>::Direction direction) const {
        assert(size_t(direction) <= dim);
        return refinements[size_t(direction)];
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param object refined object
     * \param path additional path hints pointing to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(typename Primitive<DIM>::Direction direction, const weak_ptr<const GeometryObjectD<DIM>>& object, const PathHints& path, double position) {
        auto key = std::make_pair(object, path);
        assert(size_t(direction) <= dim);
        refinements[size_t(direction)][key].insert(position);
        this->fireChanged();
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param object refined object
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(typename Primitive<DIM>::Direction direction, const weak_ptr<const GeometryObjectD<DIM>>& object, double position) {
        addRefinement(direction, object, PathHints(), position);
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param path path to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(typename Primitive<DIM>::Direction direction, const Path& path, double position) {
        addRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<DIM>>(path.back()), PathHints(path), position);
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param subtree subtree to the refined object (only the last path is used)
     * \param position position of the additional grid line in the refined object
     */
    void addRefinement(typename Primitive<DIM>::Direction direction, const GeometryObject::Subtree& subtree, double position) {
        auto path = subtree.getLastPath();
        addRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<DIM>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove refinement to the mesh
     * \param direction direction in which the object was refined
     * \param object refined object
     * \param path additional path hints pointing to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(typename Primitive<DIM>::Direction direction, const weak_ptr<const GeometryObjectD<DIM>>& object, const PathHints& path, double position) {
        auto key = std::make_pair(object, path);
        assert(size_t(direction) <= dim);
        auto ref = refinements[size_t(direction)].find(key);
        if (ref == refinements[size_t(direction)].end()) throw BadInput("RectilinearMeshDivideGenerator", "There are no refinements for specified geometry object.");
        auto oposition = ref->second.find(position);
        if (oposition == ref->second.end()) throw BadInput("RectilinearMeshDivideGenerator", "Specified geometry object does not have refinements at %1%.", *oposition);
        ref->second.erase(oposition);
        if (ref->second.empty()) refinements[size_t(direction)].erase(ref);
        this->fireChanged();
    }

    /**
     * Remove refinement to the mesh
     * \param direction direction in which the object was refined
     * \param object refined object
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(typename Primitive<DIM>::Direction direction, const weak_ptr<const GeometryObjectD<DIM>>& object, double position) {
        removeRefinement(direction, object, PathHints(), position);
    }

    /**
     * Remove refinement from the mesh
     * \param direction direction in which the object was refined
     * \param path path to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(typename Primitive<DIM>::Direction direction, const Path& path, double position) {
        removeRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<DIM>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove refinement from the mesh
     * \param direction direction in which the object was refined
     * \param subtree subtree to the refined object (only the last path is used)
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(typename Primitive<DIM>::Direction direction, const GeometryObject::Subtree& subtree, double position) {
        auto path = subtree.getLastPath();
        removeRefinement(direction, dynamic_pointer_cast<const GeometryObjectD<DIM>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove all refinements from the object
     * \param object refined object
     * \param path additional path hints pointing to the refined object
     */
    void removeRefinements(const weak_ptr<const GeometryObjectD<DIM>>& object, const PathHints& path=PathHints()) {
        auto key = std::make_pair(object, path);
        bool found;
        for (size_t i = 0; i != dim; ++i) {
            auto ref = refinements[i].find(key);
            if (ref != refinements[i].end()) {
                found = true;
                refinements[i].erase(ref);
            }
        }
        if (found) this->fireChanged();
        else writelog(LOG_WARNING, "RectilinearMeshDivideGenerator: There are no refinements for specified geometry object");
    }

    /**
     * Remove all refinements from all objects
     */
    void clearRefinements() {
        refinements[0].clear();
        refinements[1].clear();
        this->fireChanged();
    }

    /**
     * Remove all refinements from the object
     * \param path path to the refined object
     */
    void removeRefinements(const Path& path) {
        removeRefinements(dynamic_pointer_cast<const GeometryObjectD<DIM>>(path.back()), PathHints(path));
    }

    /**
     * Remove all refinements from the object
     * \param subtree subtree to the refined object (only the last path is used)
     */
    void removeRefinements(const GeometryObject::Subtree& subtree) {
        auto path = subtree.getLastPath();
        removeRefinements(dynamic_pointer_cast<const GeometryObjectD<DIM>>(path.back()), PathHints(path));
    }

};

} // namespace plask

#endif // PLASK__GENERATOR_RECTILINEAR_H
