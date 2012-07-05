#ifndef PLASK__GENERATOR_RECTILINEAR_H
#define PLASK__GENERATOR_RECTILINEAR_H

#include "mesh.h"
#include "rectilinear.h"

namespace plask {

class RectilinearMesh2DGeneratorSimple: public MeshGeneratorOf<RectilinearMesh2D> {

    size_t division;

    RectilinearMesh1D get1DMesh(const RectilinearMesh1D& initial_mesh);

  protected:
    virtual shared_ptr<RectilinearMesh2D> generate(const shared_ptr<GeometryElementD<2>>& geometry);

  public:

      RectilinearMesh2DGeneratorSimple(size_t div=1, double factor=2.0) : division(div) {}

      /// Get initial division of the smallest element in the mesh
      inline size_t getDivision() { return division; }
      /// Set initial division of the smallest element in the mesh
      inline void setDivision(size_t div) { division = div; clearCache(); }
};

} // namespace plask

#endif // PLASK__GENERATOR_RECTILINEAR_H