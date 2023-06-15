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
#ifndef PLASK__GENERATOR_RECTANGULAR_H
#define PLASK__GENERATOR_RECTANGULAR_H

#include "mesh.hpp"
#include "rectangular.hpp"
#include "plask/geometry/path.hpp"

namespace plask {

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry object
 * \param split if not 0 split points at boundaries into two separated by \p split
 * \return generated mesh
 */
PLASK_API shared_ptr<OrderedAxis> makeGeometryGrid1D(const shared_ptr<GeometryObjectD<2>>& geometry, double split=0.);

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry object
 * \param split if not 0 split points at boundaries into two separated by \p split
 * \return generated mesh
 */
PLASK_API shared_ptr<RectangularMesh<2>> makeGeometryGrid(const shared_ptr<GeometryObjectD<2>>& geometry, double split=0.);

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry object
 * \param split if not 0 split points at boundaries into two separated by \p split
 * \return generated mesh
 */
PLASK_API shared_ptr<RectangularMesh<3>> makeGeometryGrid(const shared_ptr<GeometryObjectD<3>>& geometry, double split=0.);

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry
 * \return generated mesh
 */
inline shared_ptr<OrderedAxis> makeGeometryGrid1D(const shared_ptr<GeometryD<2>>& geometry)
{
    return makeGeometryGrid1D(geometry->getChild());
}

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry
 * \return generated mesh
 */
inline shared_ptr<RectangularMesh<2>> makeGeometryGrid(const shared_ptr<GeometryD<2>>& geometry)
{
    return makeGeometryGrid(geometry->getChild());
}

/**
 * Generate grid along edges of bounding boxes of all geometry elements
 * \param geometry given geometry
 * \return generated mesh
 */
inline shared_ptr<RectangularMesh<3>> makeGeometryGrid(const shared_ptr<GeometryD<3>>& geometry)
{
    return makeGeometryGrid(geometry->getChild());
}

/**
 * Divide existing axis into finer mesh with points approximately spaced by \p spacing.
 * \param spacing approximate distance between mesh points
 * \return refined mesh
 */
PLASK_API shared_ptr<OrderedAxis> refineAxis(const shared_ptr<MeshAxis>& axis, double spacing);

/**
 * Generator of basic 2D geometry grid
 */
struct PLASK_API OrderedMesh1DSimpleGenerator: public MeshGeneratorD<1> {

  protected:

    bool split;

  public:

    /// Create generator
    OrderedMesh1DSimpleGenerator(bool split=false): split(split) {}

    shared_ptr<MeshD<1>> generate(const shared_ptr<GeometryObjectD<2>>& geometry) override;
};


/**
 * Generator of basic 2D geometry grid
 */
struct PLASK_API RectangularMesh2DSimpleGenerator: public MeshGeneratorD<2> {

  protected:

    bool split;

  public:

    /// Create generator
    RectangularMesh2DSimpleGenerator(bool split=false): split(split) {}

    shared_ptr<MeshD<2>> generate(const shared_ptr<GeometryObjectD<2>>& geometry) override;
};

/**
 * Generator of basic 3D geometry grid
 */
struct PLASK_API RectangularMesh3DSimpleGenerator: public MeshGeneratorD<3> {

  protected:

    bool split;

  public:

    /// Create generator
    RectangularMesh3DSimpleGenerator(bool split=false): split(split) {}

    shared_ptr<MeshD<3>> generate(const shared_ptr<GeometryObjectD<3>>& geometry) override;
};



/**
 * Generator of basic 2D geometry grid with approximately equal spacing.
 * This generator respects existing object boundaries.
 */
class PLASK_API OrderedMesh1DRegularGenerator: public OrderedMesh1DSimpleGenerator {

    /// Requested spacing
    double spacing;

  public:

    /**
     * Create generator
     * \param spacing approximate distance between mesh lines
     */
    OrderedMesh1DRegularGenerator(double spacing, bool split=false):
        OrderedMesh1DSimpleGenerator(split), spacing(spacing) {}

    shared_ptr<MeshD<1>> generate(const shared_ptr<GeometryObjectD<2>>& geometry) override;
};


/**
 * Generator of basic 2D geometry grid with approximately equal spacing.
 * This generator respects existing object boundaries.
 */
class PLASK_API RectangularMesh2DRegularGenerator: public RectangularMesh2DSimpleGenerator {

    /// Requested spacing
    double spacing0, spacing1;

  public:

    /**
     * Create generator
     * \param spacing approximate distance between mesh lines
     */
    RectangularMesh2DRegularGenerator(double spacing, bool split=false):
        RectangularMesh2DSimpleGenerator(split), spacing0(spacing), spacing1(spacing) {}

    /**
     * Create generator
     * \param spacing0 approximate distance between mesh lines for horizontal axis
     * \param spacing1 approximate distance between mesh lines for vertical axis
     */
    RectangularMesh2DRegularGenerator(double spacing0, double spacing1, bool split=false):
        RectangularMesh2DSimpleGenerator(split), spacing0(spacing0), spacing1(spacing1) {}

    shared_ptr<MeshD<2>> generate(const shared_ptr<GeometryObjectD<2>>& geometry) override;
};

/**
 * Generator of basic 3D geometry grid with approximately equal spacing.
 * This generator respects existing object boundaries.
 */
struct PLASK_API RectangularMesh3DRegularGenerator: public RectangularMesh3DSimpleGenerator {

    /// Requested spacing
    double spacing0, spacing1, spacing2;

  public:

    /**
     * Create generator
     * \param spacing approximate distance between mesh lines
     */
    RectangularMesh3DRegularGenerator(double spacing, bool split=false):
        RectangularMesh3DSimpleGenerator(split), spacing0(spacing), spacing1(spacing), spacing2(spacing) {}

    /**
     * Create generator
     * \param spacing0 approximate distance between mesh lines for longitudinal axis
     * \param spacing1 approximate distance between mesh lines for transverse axis
     * \param spacing2 approximate distance between mesh lines for vertical axis
     */
    RectangularMesh3DRegularGenerator(double spacing0, double spacing1, double spacing2, bool split=false):
        RectangularMesh3DSimpleGenerator(split), spacing0(spacing0), spacing1(spacing1), spacing2(spacing2) {}

    shared_ptr<MeshD<3>> generate(const shared_ptr<GeometryObjectD<3>>& geometry) override;
};



/**
 * Generator of 2D geometry grid using other generator for horizontal axis
 */
class PLASK_API RectangularMesh2DFrom1DGenerator: public MeshGeneratorD<2> {

    shared_ptr<MeshGeneratorD<1>> horizontal_generator;

  public:

    /**
     * Create generator
     */
    RectangularMesh2DFrom1DGenerator(const shared_ptr<MeshGeneratorD<1>>& source):
        horizontal_generator(source) {}

    shared_ptr<MeshD<2>> generate(const shared_ptr<GeometryObjectD<2>>& geometry) override;
};



/**
 * Dividing generator ensuring no rapid change of element size
 */
template <int dim>
struct PLASK_API RectangularMeshRefinedGenerator: public MeshGeneratorD<dim> {

    typedef typename Rectangular_t<dim>::Rectilinear GeneratedMeshType;
    using MeshGeneratorD<dim>::DIM;

    typedef std::map<std::pair<weak_ptr<const GeometryObjectD<DIM>>,PathHints>, std::set<double>> Refinements;

    double aspect;

    Refinements refinements[dim];

    shared_ptr<OrderedAxis> getAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir);

    virtual shared_ptr<OrderedAxis> processAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir) = 0;

    virtual const char* name() = 0;

    void fromXML(XMLReader&, Manager&);

    std::pair<double, double> getMinMax(const shared_ptr<OrderedAxis>& axis);

    void divideLargestSegment(shared_ptr<OrderedAxis> axis);

    /**
     * Create new generator
     */
    RectangularMeshRefinedGenerator() : aspect(0) {}

    shared_ptr<MeshD<dim>> generate(const shared_ptr<GeometryObjectD<DIM>>& geometry) override;

    /// \return true if the adjacent mesh elements cannot differ more than twice in size along each axis
    double getAspect() const { return aspect; }

    /// \param value true if the adjacent mesh elements cannot differ more than twice in size along each axis
    void setAspect(double value) {
        if (value != 0. && value < 2.)
            throw BadInput("DivideGenerator", "Maximum aspect must be larger than 2");
        aspect = value;
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
        if (ref == refinements[size_t(direction)].end()) throw BadInput("RectangularMeshDivideGenerator", "There are no refinements for specified geometry object.");
        auto oposition = ref->second.find(position);
        if (oposition == ref->second.end()) throw BadInput("RectangularMeshDivideGenerator", "Specified geometry object does not have refinements at {0}.", *oposition);
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
        bool found = false;
        for (size_t i = 0; i != dim; ++i) {
            auto ref = refinements[i].find(key);
            if (ref != refinements[i].end()) {
                found = true;
                refinements[i].erase(ref);
            }
        }
        if (found) this->fireChanged();
        else writelog(LOG_WARNING, "RectangularMeshDivideGenerator: There are no refinements for specified geometry object");
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

template <> shared_ptr<MeshD<1>> RectangularMeshRefinedGenerator<1>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry);
template <> shared_ptr<MeshD<2>> RectangularMeshRefinedGenerator<2>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry);
template <> shared_ptr<MeshD<3>> RectangularMeshRefinedGenerator<3>::generate(const boost::shared_ptr<plask::GeometryObjectD<3>>& geometry);

PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshRefinedGenerator<1>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshRefinedGenerator<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshRefinedGenerator<3>)

/**
 * Dividing generator ensuring no rapid change of element size
 */
template <int dim>
struct PLASK_API RectangularMeshDivideGenerator: public RectangularMeshRefinedGenerator<dim> {

    typedef typename Rectangular_t<dim>::Rectilinear GeneratedMeshType;
    using MeshGeneratorD<dim>::DIM;

    size_t pre_divisions[dim];
    size_t post_divisions[dim];

    char gradual;

    shared_ptr<OrderedAxis> processAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir) override;

    const char* name() override { return "DivideGenerator"; }

    template <int fd>
    friend shared_ptr<MeshGenerator> readRectangularDivideGenerator(XMLReader&, Manager&);

    /**
     * Create new generator
     */
    RectangularMeshDivideGenerator(): gradual(7)
    {
        for (int i = 0; i != dim; ++i) {
            pre_divisions[i] = 1;
            post_divisions[i] = 1;
        }
    }

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

    /// \return true if the adjacent mesh elements cannot differ more than twice in size along axis \p dir
    inline bool getGradual(size_t dir) const {
        assert(dir <= dim);
        return (gradual >> dir) & 1;
    }

    /// \param value true if the adjacent mesh elements cannot differ more than twice in size along axis \p dir
    inline void setGradual(size_t dir, bool value) {
        assert(dir <= dim);
        gradual &= ~(1 << dir);
        if (value) gradual |= 1 << dir;
        this->fireChanged();
    }

    /// \return true if the adjacent mesh elements cannot differ more than twice in size along axis \p direction
    inline bool getGradual(typename Primitive<DIM>::Direction direction) const {
        return getGradual(size_t(direction));
    }

    /// \param value true if the adjacent mesh elements cannot differ more than twice in size along axis \p direction
    inline void setGradual(typename Primitive<DIM>::Direction direction, bool value) {
        setGradual(size_t(direction), value);
    }

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshDivideGenerator<1>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshDivideGenerator<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshDivideGenerator<3>)


/**
 * Dense-edge genereator that has very dense sampling near edges and gradually gets wider towards the center.
 */
template <int dim>
struct PLASK_API RectangularMeshSmoothGenerator: public RectangularMeshRefinedGenerator<dim> {

    typedef typename Rectangular_t<dim>::Rectilinear GeneratedMeshType;
    using MeshGeneratorD<dim>::DIM;

    double finestep[dim];   ///< Small step next to the boundary
    double maxstep[dim];    ///< Maximum step in the mesh
    double factor[dim];     ///< Maximum element increase factor

    shared_ptr<OrderedAxis> processAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir) override;

    const char* name() override { return "SmoothGenerator"; }

    template <int fd>
    friend shared_ptr<MeshGenerator> readRectangularSmoothGenerator(XMLReader&, Manager&);

    /// Create new generator
    RectangularMeshSmoothGenerator();

    /// Get small step next to the boundary
    inline double getFineStep(typename Primitive<DIM>::Direction direction) const {
        assert(size_t(direction) <= dim);
        return finestep[size_t(direction)];
    }

    /// Set small step next to the boundary
    inline void setFineStep(typename Primitive<DIM>::Direction direction, double value) {
        assert(size_t(direction) <= dim);
        finestep[size_t(direction)] = value;
        this->fireChanged();
    }

    /// Get small step next to the boundary
    inline double getMaxStep(typename Primitive<DIM>::Direction direction) const {
        assert(size_t(direction) <= dim);
        return maxstep[size_t(direction)];
    }

    /// Set small step next to the boundary
    inline void setMaxStep(typename Primitive<DIM>::Direction direction, double value) {
        assert(size_t(direction) <= dim);
        maxstep[size_t(direction)] = value;
        this->fireChanged();
    }

    /// Get maximum element increase factor
    inline double getFactor(typename Primitive<DIM>::Direction direction) const {
        assert(size_t(direction) <= dim);
        return factor[size_t(direction)];
    }

    /// Set maximum element increase factor
    inline void setFactor(typename Primitive<DIM>::Direction direction, double value) {
        assert(size_t(direction) <= dim);
        if (value < 1.) throw BadInput("SmoothGenerator", "Increase factor for axis {:d} cannot be smaller than 1", size_t(direction));
        factor[size_t(direction)] = value;
        this->fireChanged();
    }
};

template<> RectangularMeshSmoothGenerator<1>::RectangularMeshSmoothGenerator();
template<> RectangularMeshSmoothGenerator<2>::RectangularMeshSmoothGenerator();
template<> RectangularMeshSmoothGenerator<3>::RectangularMeshSmoothGenerator();

PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshSmoothGenerator<1>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshSmoothGenerator<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(RectangularMeshSmoothGenerator<3>)

} // namespace plask

#endif // PLASK__GENERATOR_RECTANGULAR_H
