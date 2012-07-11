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


class RectilinearMesh2DDividingGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

    size_t divisions[2];

    typedef std::map<PathHints, std::set<double>> Refinements;

    Refinements refinements[2];

    RectilinearMesh1D get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<plask::GeometryElementD<2>>& geometry, size_t dir);

  public:

    bool warn_multiple, ///< Warn if a single refinement points to more than one object.
         warn_none,     ///< Warn if a defined refinement points to object absent from provided geometry.
         warn_outside;  ///< Warn if a defined refinemtent takes place outside of the pointed object.

    RectilinearMesh2DDividingGenerator(size_t div0=1, size_t div1=0) : warn_multiple(true), warn_outside(true) {
        divisions[0] = div0;
        divisions[1] = div1? div1 : div0;
    }

    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryElementD<2>>& geometry);

    /// Get initial division of the smallest element in the mesh
    inline std::pair<size_t,size_t> getDivision() const { return std::pair<size_t,size_t>(divisions[0], divisions[1]); }
    /// Set initial division of the smallest element in the mesh
    inline void setDivision(size_t div0, size_t div1=0) {
        divisions[0] = div0;
        divisions[1] = div1? div1 : div0;
        clearCache();
    }

    /// \return map od refinements
    /// \param direction direction of the refinements
    const Refinements& getRefinements(Primitive<2>::DIRECTION direction) const {
        return refinements[std::size_t(direction)];
    }

    /**
    * Add refinement to the mesh
    * \param path path hints pointing to the refined object
    * \param direction direction in which the object should be refined
    * \param position position of the additional grid line in the refined object
    */
    void addRefinement(const PathHints& path, Primitive<2>::DIRECTION direction, double position) {
        refinements[std::size_t(direction)][path].insert(position);
        clearCache();
    }

    /**
    * Remove refinement to the mesh
    * \param path path hints pointing to the refined object
    * \param direction direction in which the object should be refined
    * \param position position of the additional grid line in the refined object
    */
    void removeRefinement(const PathHints& path, Primitive<2>::DIRECTION direction, double position) {
        auto object = refinements[std::size_t(direction)].find(path);
        if (object == refinements[std::size_t(direction)].end()) throw BadInput("RectilinearMesh2DDividingGenerator", "There are no refinements for specified geometry element.");
        auto oposition = object->second.find(position);
        if (oposition == object->second.end()) throw BadInput("RectilinearMesh2DDividingGenerator", "Specified geometry element does not have refinements on %1%.", *oposition);
        object->second.erase(oposition);
        if (object->second.empty()) refinements[std::size_t(direction)].erase(object);
        clearCache();
    }

    /**
    * Remove refinements all refinements from the object
    * \param path path hints pointing to the refined object
    * \param direction direction in which the object should be refined
    * \param position position of the additional grid line in the refined object
    */
    void removeRefinements(const PathHints& path) {
        auto object0 = refinements[0].find(path);
        auto object1 = refinements[1].find(path);

        if (object0 == refinements[0].end() && object1 == refinements[1].end())
            throw BadInput("RectilinearMesh2DDividingGenerator", "There are no refinements for specified geometry element.");
        else {
            if (object0 != refinements[0].end()) refinements[0].erase(object0);
            if (object1 != refinements[1].end()) refinements[1].erase(object1);
            clearCache();
        }
    }
};

} // namespace plask

#endif // PLASK__GENERATOR_RECTILINEAR_H