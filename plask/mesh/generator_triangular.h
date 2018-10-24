#ifndef PLASK__GENERATOR_TRIANGULAR_H
#define PLASK__GENERATOR_TRIANGULAR_H

#include "mesh.h"

namespace plask {

/**
 * Generator which creates triangular mesh by Triangle library authored by Jonathan Richard Shewchuk.
 *
 * Triangle generates exact Delaunay triangulations, constrained Delaunay triangulations,
 * conforming Delaunay triangulations, Voronoi diagrams, and high-quality triangular meshes.
 * The latter can be generated with no small or large angles,
 * and are thus suitable for finite element analysis.
 *
 * See: http://www.cs.cmu.edu/%7Equake/triangle.html
 */
struct TriangleGenerator: public MeshGeneratorD<2> {

    shared_ptr<MeshD<2>> generate(const shared_ptr<GeometryObjectD<DIM>>& geometry) override;

private:

    /**
     * Get switches for triangulate function of Triangle.
     * @return the switches
     */
    std::string getSwitches() const;
};

}   // namespace plask

#endif // PLASK__GENERATOR_TRIANGULAR_H
