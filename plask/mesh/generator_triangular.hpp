#ifndef PLASK__GENERATOR_TRIANGULAR_H
#define PLASK__GENERATOR_TRIANGULAR_H

#include "mesh.hpp"

namespace plask {

/**
 * Generator which creates triangular mesh by Triangle library authored by Jonathan Richard Shewchuk.
 *
 * Triangle generates exact Delaunay triangulations, constrained Delaunay triangulations,
 * conforming Delaunay triangulations, Voronoi diagrams, and high-quality triangular meshes.
 * The latter can be generated with no small or large angles,
 * and are thus suitable for finite element analysis.
 *
 * See: https://www.cs.cmu.edu/~quake/triangle.html
 */
struct PLASK_API TriangleGenerator: public MeshGeneratorD<2> {

    shared_ptr<MeshD<2>> generate(const shared_ptr<GeometryObjectD<DIM>>& geometry) override;

    /**
     * A maximum triangle area constraint.
     */
    optional<double> maxTriangleArea;

    /// Set maximum triangle area
    void setMaxTriangleArea(double area) {
        maxTriangleArea.reset(area);
        fireChanged();
    }

    /// Clear maximum triangle area
    void clearMaxTriangleArea() {
        maxTriangleArea.reset();
        fireChanged();
    }

    /**
     * Minimum angle (if this is NaN, Triangle default is used, 20 degrees).
     */
    optional<double> minTriangleAngle;

    /// Set minimum angle
    void setMinTriangleAngle(double angle) {
        minTriangleAngle.reset(angle);
        fireChanged();
    }

    /// Clear minimum angle
    void clearMinTriangleAngle() {
        minTriangleAngle.reset();
        fireChanged();
    }

    /**
     * Use true Delaunay triangulation
     */
    bool delaunay = true;

    /**
     * Use full mesh
     */
    bool full = false;

    /// Set full
    void setFull(bool value) {
        full = value;
        fireChanged();
    }

private:

    /**
     * Get switches for triangulate function of Triangle.
     * @return the switches
     */
    std::string getSwitches() const;


};

}   // namespace plask

#endif // PLASK__GENERATOR_TRIANGULAR_H
