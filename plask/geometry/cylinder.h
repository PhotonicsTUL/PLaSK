#ifndef PLASK__GEOMETRY_CYLINDER_H
#define PLASK__GEOMETRY_CYLINDER_H

#include "leaf.h"

namespace plask {

/**
 * Cylinder with given height and radius of base.
 *
 * Center of cylinders' base lies in point (0.0, 0.0, 0.0)
 */
struct Cylinder: public GeometryObjectLeaf<3> {

    double radius, height;

    static constexpr const char* NAME = "cylinder";

    virtual std::string getTypeName() const { return NAME; }

    Cylinder(double radius, double height, const shared_ptr<Material>& material = shared_ptr<Material>());

    virtual Box getBoundingBox() const;

    virtual bool contains(const DVec& p) const;

    //virtual bool intersects(const Box& area) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

    /**
     * Set radius and inform observers about changes.
     * @param new_radius new radius to set
     */
    void setRadius(double new_radius) {
        this->radius = new_radius;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set height and inform observers about changes.
     * @param new_height new height to set
     */
    void setHeight(double new_height) {
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

    bool isSolidInBB(Primitive<3>::Direction direction) const override {
        return direction != Primitive<3>::DIRECTION_TRAN && materialProvider->isSolidInBB(direction);
    }

};

} // namespace plask

#endif // PLASK__GEOMETRY_CYLINDER_H
