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
#ifndef PLASK__GEOMETRY_PRISM_H
#define PLASK__GEOMETRY_PRISM_H

/** @file
This file contains triangle (geometry object) class.
*/

#include "../vector/lateral.hpp"

#include "leaf.hpp"

namespace plask {

/**
 * Represents prism with triangular base one vertex at point (0, 0, 0) and height h.
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API TriangularPrism : public GeometryObjectLeaf<3> {
    typedef GeometryObjectLeaf<3> BaseClass;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// 2D vector for defining base triangle
    typedef LateralVec<double> Vec2;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    static const char* NAME;

    std::string getTypeName() const override;

    /**
     * Construct a solid triangle with vertices at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle vertices
     * @param material material inside the whole triangle
     */
    explicit TriangularPrism(const Vec2& p0 = Primitive<2>::ZERO_VEC,
                   const Vec2& p1 = Primitive<2>::ZERO_VEC,
                   double height = 0.,
                   const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Construct a triangle with vertices at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle vertices
     * @param materialTopBottom describes materials inside the triangle
     */
    explicit TriangularPrism(const Vec2& p0,
                   const Vec2& p1,
                   double height,
                   shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    // explicit Prism(const DVec& p0, const Vec2& p1, double height,
    //                const std::unique_ptr<MaterialProvider>& materialProvider);

    Box3D getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<TriangularPrism>(*this); }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    /// Triangular base forming vectors
    Vec2 p0, p1;

    /// Prism height
    double height;

    /**
     * Set coordinates of first vertex and inform observers about changes.
     * \param new_p0 new coordinates for p0
     */
    void setP0(const Vec2& new_p0) {
        p0 = new_p0;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set coordinates of second vertex and inform observers about changes.
     * \param new_p0 new coordinates for p1
     */
    void setP1(const Vec2& new_p1) {
        p1 = new_p1;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set the height inform observers about changes.
     * \param new_height new height
     */
    void setHeight(double new_height) {
        height = new_height;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct PLASK_API Prism : public GeometryObjectLeaf<3> {
  protected:
    double height;

    std::vector<LateralVec<double>> vertices;

    friend shared_ptr<GeometryObject> read_prism(GeometryReader& reader);

  public:
    typedef GeometryObjectLeaf<3> BaseClass;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    static const char* NAME;

    std::string getTypeName() const override;

    /**
     * Construct an empty prism.
     */
    Prism() = default;

    /**
     * Construct a prism with vertices at specified points.
     * \param height height of the prism
     * \param vertices coordinates of the prism vertices
     * \param material material of the constructed prism
     */
    explicit Prism(double height,
                       const std::vector<LateralVec<double>>& vertices,
                       const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Construct a prism with vertices at specified points.
     * \param height height of the prism
     * \param vertices coordinates of the prism vertices
     * \param material material of the constructed prism
     */
    explicit Prism(double height,
                       std::vector<LateralVec<double>>&& vertices,
                       const shared_ptr<Material>&& material = shared_ptr<Material>());

    /**
     * Construct a prism with vertices at specified points.
     * \param height height of the prism
     * \param vertices coordinates of the prism vertices
     * \param material material of the constructed prism
     */
    explicit Prism(double height,
                       std::initializer_list<LateralVec<double>> vertices,
                       const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Construct a prism with vertices at specified points.
     * \param height height of the prism
     * \param vertices coordinates of the prism vertices
     * \param materialTopBottom materials of the constructed prism
     */
    explicit Prism(double height,
                       const std::vector<LateralVec<double>>& vertices,
                       shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    /**
     * Construct a prism with vertices at specified points.
     * \param height height of the prism
     * \param vertices coordinates of the prism vertices
     * \param materialTopBottom materials of the constructed prism
     */
    explicit Prism(double height,
                       std::vector<LateralVec<double>>&& vertices,
                       shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    /**
     * Construct a prism with vertices at specified points.
     * \param height height of the prism
     * \param vertices coordinates of the prism vertices
     * \param materialTopBottom materials of the constructed prism
     */
    explicit Prism(double height,
                       std::initializer_list<LateralVec<double>> vertices,
                       shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    /**
     * Copy-constructor
     */
    explicit Prism(const Prism& src) : BaseClass(src), height(src.height), vertices(src.vertices) {}

    /**
     * Get the prism height.
     * \return prism height
     */
    double getHeight() const { return height; }

    /**
     * Set the prism height.
     * \param height new prism height
     */
    void setHeight(double height) {
        this->height = height;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Get the prism vertices.
     * \return vector of prism vertices
     */
    const std::vector<LateralVec<double>>& getVertices() const { return vertices; }

    /**
     * Get the prism vertices.
     * \return vector of prism vertices
     */
    std::vector<LateralVec<double>>& getVertices() { return vertices; }

    /**
     * Set the prism vertices.
     * \return vector of prism vertices
     */
    void setVertices(const std::vector<LateralVec<double>>& vertices) {
        this->vertices = vertices;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set the prism vertices.
     * \return vector of prism vertices
     */
    void setVertices(std::vector<LateralVec<double>>&& vertices) {
        this->vertices = std::move(vertices);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set one prism vertex.
     * \param index index of the vertex to set
     * \param vertex new vertex to set
     */
    void setVertex(unsigned index, const LateralVec<double>& vertex) {
        vertices.at(index) = vertex;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Insert a new vertex to the prism.
     * \param index index in the vertex list where the new vertex should be inserted
     * \param vertex new vertex to insert
     */
    void insertVertex(unsigned index, const LateralVec<double>& vertex) {
        vertices.insert(vertices.begin() + index, vertex);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Add a new vertex to the prism.
     * \param vertex new vertex to insert
     */
    void addVertex(const LateralVec<double>& vertex) {
        vertices.push_back(vertex);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Remove a vertex from the prism.
     * \param index index in the vertex list of the vertex to remove
     */
    void removeVertex(unsigned index) {
        vertices.erase(vertices.begin() + index);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Clear all vertices from the prism.
     */
    void clearVertices() {
        vertices.clear();
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Get the number of vertices in the prism.
     * \return number of vertices in the prism
     */
    unsigned getVertexCount() const { return vertices.size(); }

    /**
     * Get the vertex at specified index.
     * \param index index of the vertex to get
     * \return vertex at specified index
     */
    const LateralVec<double>& getVertex(unsigned index) const { return vertices.at(index); }

    // /// Validate the prism: check if the lines between vertices do not intersect.
    // bool checkSegments() const;

    void validate() const override;

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Prism>(*this); }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

}  // namespace plask

#endif  // PLASK__GEOMETRY_PRISM_H
