#ifndef PLASK__GENERATOR_RECTILINEAR_H
#define PLASK__GENERATOR_RECTILINEAR_H

#include "mesh.h"
#include "rectilinear.h"
#include <plask/geometry/path.h>

namespace plask {

class RectilinearMesh2DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

  public:

    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryElementD<2>>& geometry);
};


class RectilinearMesh3DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh3D> {

  public:

    virtual shared_ptr<RectilinearMesh3D> generate(const shared_ptr<GeometryElementD<3>>& geometry);
};


class RectilinearMesh2DDivideGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

    size_t pre_divisions[2];
    size_t post_divisions[2];

    typedef std::map<std::pair<weak_ptr<const GeometryElementD<2>>,PathHints>, std::set<double>> Refinements;

    Refinements refinements[2];

    RectilinearMesh1D get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<plask::GeometryElementD<2>>& geometry, size_t dir);

  public:

    bool limit_change;  ///< Limit the change of size of adjacent elements to the factor of two
    bool warn_multiple, ///< Warn if a single refinement points to more than one object.
         warn_none,     ///< Warn if a defined refinement points to object absent from provided geometry.
         warn_outside;  ///< Warn if a defined refinement takes place outside of the pointed object.

    /**
     * Create new generator
     * \param prediv0 Initial mesh division in horizontal direction
     * \param postdiv0 Final mesh division in horizontal direction
     * \param prediv1 Initial mesh division in vertical direction (0 means the same as horizontal)
     * \param postdiv1 Final mesh division in vertical direction (0 means the same as horizontal)
    **/
    RectilinearMesh2DDivideGenerator(size_t prediv0=1, size_t postdiv0=1, size_t prediv1=0, size_t postdiv1=0) :
        limit_change(true), warn_multiple(true), warn_outside(true)
    {
        pre_divisions[0] = prediv0;
        pre_divisions[1] = prediv1? prediv1 : prediv0;
        post_divisions[0] = postdiv0;
        post_divisions[1] = postdiv1? postdiv1 : postdiv0;
    }

    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryElementD<2>>& geometry);

    /// Get initial division of the smallest element in the mesh
    inline std::pair<size_t,size_t> getPreDivision() const { return std::pair<size_t,size_t>(pre_divisions[0], pre_divisions[1]); }
    /// Set initial division of the smallest element in the mesh
    inline void setPreDivision(size_t div0, size_t div1=0) {
        pre_divisions[0] = div0;
        pre_divisions[1] = div1? div1 : div0;
        clearCache();
    }

    /// Get final division of the smallest element in the mesh
    inline std::pair<size_t,size_t> getPostDivision() const { return std::pair<size_t,size_t>(post_divisions[0], post_divisions[1]); }
    /// Set final division of the smallest element in the mesh
    inline void setPostDivision(size_t div0, size_t div1=0) {
        post_divisions[0] = div0;
        post_divisions[1] = div1? div1 : div0;
        clearCache();
    }

    /// \return map of refinements
    /// \param direction direction of the refinements
    const Refinements& getRefinements(Primitive<2>::DIRECTION direction) const {
        return refinements[std::size_t(direction)];
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param element refined object
     * \param position position of the additional grid line in the refined object
     * \param path additional path hints pointing to the refined object
     */
    void addRefinement(Primitive<2>::DIRECTION direction, const weak_ptr<const GeometryElementD<2>>& element, const PathHints& path, double position) {
        auto key = std::make_pair(element, path);
        refinements[std::size_t(direction)][key].insert(position);
        clearCache();
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param element refined object
     * \param position position of the additional grid line in the refined object
     * \param path additional path hints pointing to the refined object
     */
    void addRefinement(Primitive<2>::DIRECTION direction, const weak_ptr<const GeometryElementD<2>>& element, double position) {
        addRefinement(direction, element, PathHints(), position);
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param path path to the refined object
     */
    void addRefinement(Primitive<2>::DIRECTION direction, const Path& path, double position) {
        addRefinement(direction, dynamic_pointer_cast<const GeometryElementD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Add refinement to the mesh
     * \param direction direction in which the object should be refined
     * \param subtree subtree to the refined object (only the last path is used)
     */
    void addRefinement(Primitive<2>::DIRECTION direction, const GeometryElement::Subtree& subtree, double position) {
        auto path = subtree.getLastPath();
        addRefinement(direction, dynamic_pointer_cast<const GeometryElementD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove refinement to the mesh
     * \param direction direction in which the object was refined
     * \param element refined object
     * \param position position of the additional grid line in the refined object
     * \param path additional path hints pointing to the refined object
     */
    void removeRefinement(Primitive<2>::DIRECTION direction, const weak_ptr<const GeometryElementD<2>>& element, const PathHints& path, double position) {
        auto key = std::make_pair(element, path);
        auto object = refinements[std::size_t(direction)].find(key);
        if (object == refinements[std::size_t(direction)].end()) throw BadInput("RectilinearMesh2DDivideGenerator", "There are no refinements for specified geometry element.");
        auto oposition = object->second.find(position);
        if (oposition == object->second.end()) throw BadInput("RectilinearMesh2DDivideGenerator", "Specified geometry element does not have refinements at %1%.", *oposition);
        object->second.erase(oposition);
        if (object->second.empty()) refinements[std::size_t(direction)].erase(object);
        clearCache();
    }

    /**
     * Remove refinement to the mesh
     * \param direction direction in which the object was refined
     * \param element refined object
     * \param position position of the additional grid line in the refined object
     * \param path additional path hints pointing to the refined object
     */
    void removeRefinement(Primitive<2>::DIRECTION direction, const weak_ptr<const GeometryElementD<2>>& element, double position) {
        removeRefinement(direction, element, PathHints(), position);
    }

    /**
     * Remove refinement from the mesh
     * \param direction direction in which the object was refined
     * \param path path to the refined object
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(Primitive<2>::DIRECTION direction, const Path& path, double position) {
        removeRefinement(direction, dynamic_pointer_cast<const GeometryElementD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove refinement from the mesh
     * \param direction direction in which the object was refined
     * \param subtree subtree to the refined object (only the last path is used)
     * \param position position of the additional grid line in the refined object
     */
    void removeRefinement(Primitive<2>::DIRECTION direction, const GeometryElement::Subtree& subtree, double position) {
        auto path = subtree.getLastPath();
        removeRefinement(direction, dynamic_pointer_cast<const GeometryElementD<2>>(path.back()), PathHints(path), position);
    }

    /**
     * Remove all refinements from the object
     * \param element refined object
     * \param path additional path hints pointing to the refined object
     */
    void removeRefinements(const weak_ptr<const GeometryElementD<2>>& element, const PathHints& path=PathHints()) {
        auto key = std::make_pair(element, path);
        auto object0 = refinements[0].find(key);
        auto object1 = refinements[1].find(key);
        if (object0 == refinements[0].end() && object1 == refinements[1].end())
            throw BadInput("RectilinearMesh2DDivideGenerator", "There are no refinements for specified geometry element.");
        else {
            if (object0 != refinements[0].end()) refinements[0].erase(object0);
            if (object1 != refinements[1].end()) refinements[1].erase(object1);
            clearCache();
        }
    }

    /**
     * Remove all refinements from the object
     * \param direction direction in which the object was refined
     * \param path path to the refined object
     */
    void removeRefinements(const Path& path) {
        removeRefinements(dynamic_pointer_cast<const GeometryElementD<2>>(path.back()), PathHints(path));
    }

    /**
     * Remove all refinements from the object
     * \param direction direction in which the object was refined
     * \param subtree subtree to the refined object (only the last path is used)
     */
    void removeRefinements(const GeometryElement::Subtree& subtree) {
        auto path = subtree.getLastPath();
        removeRefinements(dynamic_pointer_cast<const GeometryElementD<2>>(path.back()), PathHints(path));
    }

};

} // namespace plask

#endif // PLASK__GENERATOR_RECTILINEAR_H
