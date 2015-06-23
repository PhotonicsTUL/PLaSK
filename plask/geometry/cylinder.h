#ifndef PLASK__GEOMETRY_CYLINDER_H
#define PLASK__GEOMETRY_CYLINDER_H

#include "leaf.h"

namespace plask {

/**
 * Cylinder with given height and radius of base.
 *
 * Center of cylinders' base lies in point (0.0, 0.0, 0.0)
 */
struct PLASK_API Cylinder: public GeometryObjectLeaf<3> {

    double radius, height;

    static constexpr const char* NAME = "cylinder";

    virtual std::string getTypeName() const { return NAME; }

    Cylinder(double radius, double height, const shared_ptr<Material>& material = shared_ptr<Material>());

    Cylinder(double radius, double height, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    virtual Box getBoundingBox() const;

    virtual bool contains(const DVec& p) const;

    //virtual bool intersects(const Box& area) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

    /**
     * Set radius and inform observers about changes.
     * @param new_radius new radius to set
     */
    void setRadius(double new_radius) {
        if (new_radius < 0.) new_radius = 0.;
        this->radius = new_radius;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set height and inform observers about changes.
     * @param new_height new height to set
     */
    void setHeight(double new_height) {
        if (new_height < 0.) new_height = 0.;
        this->height = new_height;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set radius and height and inform observers about changes.
     * @param radius new radius to set
     * @param height new height to set
     */
    void resize(double radius, double height) {
        this->radius = radius;
        this->height = height;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    bool isUniform(Primitive<3>::Direction direction) const override;

};

} // namespace plask

#endif // PLASK__GEOMETRY_CYLINDER_H
