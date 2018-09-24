#ifndef PLASK__GEOMETRY_PRISM_H
#define PLASK__GEOMETRY_PRISM_H

/** @file
This file contains triangle (geometry object) class.
*/

#include "leaf.h"

namespace plask {

/**
 * Represents prism with triangular base one vertex at point (0, 0, 0) and height h.
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API Prism: public GeometryObjectLeaf<3> {

    typedef GeometryObjectLeaf<3> BaseClass;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// 2D vector for defining base triangle
    typedef Vec<2> Vec2;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    static const char* NAME;

    virtual std::string getTypeName() const override;

    /**
     * Contruct a solid triangle with vertexes at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle vertexes
     * @param material material inside the whole triangle
     */
    explicit Prism(const Vec2& p0 = Primitive<2>::ZERO_VEC, const Vec2& p1 = Primitive<2>::ZERO_VEC, double height=0.,
                   const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Contruct a triangle with vertexes at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle vertexes
     * @param materialTopBottom describes materials inside the triangle
     */
    explicit Prism(const Vec2& p0, const Vec2& p1, double height,
                   shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    // explicit Prism(const DVec& p0, const Vec2& p1, double height,
    //                const std::unique_ptr<MaterialProvider>& materialProvider);

    virtual Box3D getBoundingBox() const override;

    virtual bool contains(const DVec& p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override {
        return make_shared<Prism>(*this);
    }

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    bool isUniform(Primitive<3>::Direction direction) const override;

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

}   // namespace plask

#endif // PLASK__GEOMETRY_PRISM_H
