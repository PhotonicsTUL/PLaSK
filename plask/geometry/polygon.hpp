/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2024 Lodz University of Technology
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
#ifndef PLASK_GEOMETRY_POLYGONS_HPP
#define PLASK_GEOMETRY_POLYGONS_HPP

/** \file
 * This file contains polygon (geometry object) class.
 */

#include "leaf.hpp"

namespace plask {

struct PLASK_API Polygon : public GeometryObjectLeaf<2> {
  protected:
    std::vector<Vec<2>> vertices;

    friend shared_ptr<GeometryObject> readPolygon(GeometryReader& reader);

  public:
    typedef GeometryObjectLeaf<2> BaseClass;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    static const char* NAME;

    std::string getTypeName() const override;

    /**
     * Construct an empty polygon.
     */
    Polygon() = default;

    /**
     * Construct a polygon with vertices at specified points.
     * \param vertices coordinates of the polygon vertices
     * \param material material of the constructed polygon
     */
    explicit Polygon(const std::vector<Vec<2>>& vertices, const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Construct a polygon with vertices at specified points.
     * \param vertices coordinates of the polygon vertices
     * \param material material of the constructed polygon
     */
    explicit Polygon(std::vector<Vec<2>>&& vertices, const shared_ptr<Material>&& material = shared_ptr<Material>());

    /**
     * Construct a polygon with vertices at specified points.
     * \param vertices coordinates of the polygon vertices
     * \param material material of the constructed polygon
     */
    explicit Polygon(std::initializer_list<Vec<2>> vertices, const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Construct a polygon with vertices at specified points.
     * \param vertices coordinates of the polygon vertices
     * \param materialTopBottom materials of the constructed polygon
     */
    explicit Polygon(const std::vector<Vec<2>>& vertices, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    /**
     * Construct a polygon with vertices at specified points.
     * \param vertices coordinates of the polygon vertices
     * \param materialTopBottom materials of the constructed polygon
     */
    explicit Polygon(std::vector<Vec<2>>&& vertices, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    /**
     * Construct a polygon with vertices at specified points.
     * \param vertices coordinates of the polygon vertices
     * \param materialTopBottom materials of the constructed polygon
     */
    explicit Polygon(std::initializer_list<Vec<2>> vertices, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    /**
     * Copy-constructor
     */
    explicit Polygon(const Polygon& src) : BaseClass(src), vertices(src.vertices) {}

    /**
     * Get the polygon vertices.
     * \return vector of polygon vertices
     */
    const std::vector<Vec<2>>& getVertices() const { return vertices; }

    /**
     * Get the polygon vertices.
     * \return vector of polygon vertices
     */
    std::vector<Vec<2>>& getVertices() { return vertices; }

    /**
     * Set the polygon vertices.
     * \return vector of polygon vertices
     */
    void setVertices(const std::vector<Vec<2>>& vertices) {
        this->vertices = vertices;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set the polygon vertices.
     * \return vector of polygon vertices
     */
    void setVertices(std::vector<Vec<2>>&& vertices) {
        this->vertices = std::move(vertices);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set one polygon vertex.
     * \param index index of the vertex to set
     * \param vertex new vertex to set
     */
    void setVertex(unsigned index, const Vec<2>& vertex) {
        vertices.at(index) = vertex;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Insert a new vertex to the polygon.
     * \param index index in the vertex list where the new vertex should be inserted
     * \param vertex new vertex to insert
     */
    void insertVertex(unsigned index, const Vec<2>& vertex) {
        vertices.insert(vertices.begin() + index, vertex);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Add a new vertex to the polygon.
     * \param vertex new vertex to insert
     */
    void addVertex(const Vec<2>& vertex) {
        vertices.push_back(vertex);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Remove a vertex from the polygon.
     * \param index index in the vertex list of the vertex to remove
     */
    void removeVertex(unsigned index) {
        vertices.erase(vertices.begin() + index);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Clear all vertices from the polygon.
     */
    void clearVertices() {
        vertices.clear();
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Get the number of vertices in the polygon.
     * \return number of vertices in the polygon
     */
    unsigned getVertexCount() const { return vertices.size(); }

    /**
     * Get the vertex at specified index.
     * \param index index of the vertex to get
     * \return vertex at specified index
     */
    const Vec<2>& getVertex(unsigned index) const { return vertices.at(index); }

    // /// Validate the polygon: check if the lines between vertices do not intersect.
    // bool checkSegments() const;

    void validate() const override;

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Polygon>(*this); }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

}  // namespace plask

#endif
