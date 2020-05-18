#ifndef PLASK__GEOMETRY_SEPARATOR_H
#define PLASK__GEOMETRY_SEPARATOR_H

/** @file
This file contains geometry objects separators classes.
*/

#include "object.h"

namespace plask {

/**
 * Template of base classes for all separator nodes.
 *
 * Separator nodes are mainly used internally by other geometry objects and are hidden in geometry graph:
 * - return empty child/leafs/bouding box vectors, etc.
 * - return nullptr as material
 * Separator only have bouding box.
 * You should never construct separators classes directly.
 * @tparam dim number of dimensions
 * @ingroup GEOMETRY_OBJ
 */
template <int dim> struct PLASK_API GeometryObjectSeparator : public GeometryObjectD<dim> {
    typedef typename GeometryObjectD<dim>::DVec DVec;
    typedef typename GeometryObjectD<dim>::Box Box;
    using GeometryObjectD<dim>::getBoundingBox;
    using GeometryObjectD<dim>::shared_from_this;

    GeometryObject::Type getType() const override;

    static const char* NAME;

    std::string getTypeName() const override;

    /**
     * Separators typically have no materials, so this just return nullptr.
     * @param p point
     * @return @c nullptr
     */
    shared_ptr<Material> getMaterial(const DVec& p) const override;

    // void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryObject>, Box, DVec>>& dest, const
    // PathHints* path = 0) const override;

    void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                               std::vector<Box>& dest,
                               const PathHints* path = 0) const override;

    void getObjectsToVec(const GeometryObject::Predicate& predicate,
                         std::vector<shared_ptr<const GeometryObject>>& dest,
                         const PathHints* path = 0) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate,
                           std::vector<DVec>& dest,
                           const PathHints* = 0) const override;

    /*    inline void getLeafsToVec(std::vector< shared_ptr<const GeometryObject> >& dest) const override {
            dest.push_back(this->shared_from_this());
        }

        inline std::vector< shared_ptr<const GeometryObject> > getLeafs() const override {
            return { this->shared_from_this() };
        }*/

    bool hasInSubtree(const GeometryObject& el) const override;

    GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const override;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool = false) const override;

    std::size_t getChildrenCount() const override { return 0; }

    shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override;

    shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer,
                                                    Vec<3, double>* translation = 0) const override;

    // void extractToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObjectD<dim>
    // > >& dest, const PathHints* = 0) const {
    //     if (predicate(*this)) dest.push_back(static_pointer_cast< const GeometryObjectD<dim>
    //     >(this->shared_from_this()));
    // }

    shared_ptr<GeometryObject> deepCopy(
        std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const override;

    bool contains(const DVec& p) const override;

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override {}

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override {}

    /*bool intersects(const Box& area) const override {
        return this->getBoundingBox().intersects(area); //TODO ?? maybe set area to empty
    }*/
};

PLASK_API_EXTERN_TEMPLATE_STRUCT(GeometryObjectSeparator<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(GeometryObjectSeparator<3>)

/**
 * Gap in one, choose at compile time, direction.
 * @tparam dim number of dimensions
 * @tparam direction direction of gap, from 0 to dim-1
 * @ingroup GEOMETRY_OBJ
 */
template <int dim, int direction> struct Gap1D : public GeometryObjectSeparator<dim> {
    static_assert(direction < dim, "direction must be from 0 to dim-1");

    typedef GeometryObjectSeparator<dim> BaseClass;

    typedef typename BaseClass::DVec DVec;
    typedef typename BaseClass::Box Box;

    static constexpr const char* NAME = "gap";            ///< name of gap type, used as XML tag name when write to XML
    static constexpr const char* XML_SIZE_ATTR = "size";  ///< name of size attribute in XML

    std::string getTypeName() const override { return NAME; }

    /// Size of gap.
    double size;

    /**
     * Construct gap with given size.
     * @param size size of gap
     */
    Gap1D(double size = 0.0) : size(size) {}

    Box getBoundingBox() const override {
        auto size_vec = Primitive<dim>::ZERO_VEC;
        size_vec[direction] = size;
        return Box(Primitive<dim>::ZERO_VEC, size_vec);
    }

    /**
     * Set length and inform observers about changes.
     * @param new_size new length to set
     */
    void setSize(double new_size) {
        size = new_size;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override {
        BaseClass::writeXMLAttr(dest_xml_object, axes);
        dest_xml_object.attr(XML_SIZE_ATTR, size);
    }

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Gap1D>(size); }
};

}  // namespace plask

#endif  // PLASK__GEOMETRY_SEPARATOR_H
