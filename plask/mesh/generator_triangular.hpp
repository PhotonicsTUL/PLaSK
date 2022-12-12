/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
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
