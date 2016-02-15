#ifndef PLASK__GEOMETRY_CIRCLE_H
#define PLASK__GEOMETRY_CIRCLE_H

/** @file
This file contains circle (geometry object) class.
*/

#include "leaf.h"

namespace plask {

/**
 * Represents circle (sphere in 3D) with given radius and center at point (0, 0).
 * @ingroup GEOMETRY_OBJ
 */
template <int dim>
struct Circle: public GeometryObjectLeaf<dim> {

    double radius;  ///< radius of this circle

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::Box Box;

    static const char* NAME;

    virtual std::string getTypeName() const override;

    explicit Circle(double radius, const shared_ptr<Material>& material = shared_ptr<Material>());

    explicit Circle(double radius, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom): GeometryObjectLeaf<dim>(materialTopBottom), radius(radius) {
        if (radius < 0.) radius = 0.;
    }

    virtual Box getBoundingBox() const override;

    virtual bool contains(const DVec& p) const override;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    bool isUniform(Primitive<3>::Direction direction) const override;

    /**
     * Set radius and inform observers about changes.
     * @param radius new radius
     */
    void setRadius(double new_radius) {
        if (new_radius < 0.) new_radius = 0.;
        this->radius = new_radius;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }
};

template <> typename Circle<2>::Box Circle<2>::getBoundingBox() const;
template <> typename Circle<3>::Box Circle<3>::getBoundingBox() const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(Circle<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Circle<3>)

}   // namespace plask

#endif // PLASK__GEOMETRY_CIRCLE_H
