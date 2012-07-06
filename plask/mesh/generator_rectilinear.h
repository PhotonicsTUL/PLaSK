#ifndef PLASK__GENERATOR_RECTILINEAR_H
#define PLASK__GENERATOR_RECTILINEAR_H

#include "mesh.h"
#include "rectilinear.h"
#include <plask/geometry/path.h>

namespace plask {

class RectilinearMesh2DSimpleGenerator: public MeshGeneratorOf<RectilinearMesh2D> {

    size_t division;

    typedef std::map<PathHints, std::set<double>> Refinements;

    Refinements refinements[2];

    RectilinearMesh1D get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<plask::GeometryElementD<2>>& geometry, size_t dir);

  protected:
    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryElementD<2>>& geometry);

  public:

    enum Direction {
        HORIZONTAL = 0,
        VERTICAL = 1
    };

    RectilinearMesh2DSimpleGenerator(size_t div=1) : division(div) {}

    /// Get initial division of the smallest element in the mesh
    inline size_t getDivision() { return division; }
    /// Set initial division of the smallest element in the mesh
    inline void setDivision(size_t div) { division = div; clearCache(); }

    /**
    * Add refinement to the mesh
    * \param path path hints pointing to the refined object
    * \param direction direction in which the object should be refined
    * \param position position of the additional grid line in the refined object
    */
    void addRefinement(const PathHints& path, Direction direction, double position) {
        refinements[std::size_t(direction)][path].insert(position);
    }

    /**
    * Remove refinement to the mesh
    * \param path path hints pointing to the refined object
    * \param direction direction in which the object should be refined
    * \param position position of the additional grid line in the refined object
    */
    void removeRefinement(const PathHints& path, Direction direction, double position) {
        auto object = refinements[std::size_t(direction)].find(path);
        if (object == refinements[std::size_t(direction)].end()) throw BadInput("RectilinearMesh2DSimpleGenerator", "There are no refinements for specified geometry element.");
        auto oposition = object->second.find(position);
        if (oposition == object->second.end()) throw BadInput("RectilinearMesh2DSimpleGenerator", "Specified geometry element does not have refinements on %1%.", *oposition);
        object->second.erase(oposition);
        if (object->second.empty()) refinements[std::size_t(direction)].erase(object);
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

        if (object0 != refinements[0].end()) refinements[0].erase(object0);
        if (object1 != refinements[1].end()) refinements[1].erase(object1);
        if (object0 == refinements[0].end() && object1 == refinements[1].end())
            throw BadInput("RectilinearMesh2DSimpleGenerator", "There are no refinements for specified geometry element.");
    }
};

} // namespace plask

#endif // PLASK__GENERATOR_RECTILINEAR_H