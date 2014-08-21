#ifndef TRIANGLE_H
#define TRIANGLE_H

/** @file
This file contains triangle (geometry object) class.
*/

#include "leaf.h"

namespace plask {

/**
 * Represents triangle with one corner at point (0, 0).
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API Triangle: public GeometryObjectLeaf<2> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<2>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<2>::Box Box;

    static constexpr const char* NAME = "triangle";

    virtual std::string getTypeName() const override;

    /**
     * Contruct a solid triangle with corners at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle corners
     * @param material material inside the whole triangle
     */
    explicit Triangle(const DVec& p0 = Primitive<2>::ZERO_VEC, const DVec& p1 = Primitive<2>::ZERO_VEC, const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Contruct a triangle with corners at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle corners
     * @param materialTopBottom describes materials inside the triangle
     */
    explicit Triangle(const DVec& p0, const DVec& p1, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    virtual Box2D getBoundingBox() const override;

    virtual bool contains(const DVec& p) const override;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    DVec p0, p1;
};

}   // namespace plask

#endif // TRIANGLE_H
