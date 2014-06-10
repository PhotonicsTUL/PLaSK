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
template < int dim >
struct GeometryObjectSeparator: public GeometryObjectD<dim> {

    typedef typename GeometryObjectD<dim>::DVec DVec;
    typedef typename GeometryObjectD<dim>::Box Box;
    using GeometryObjectD<dim>::getBoundingBox;
    using GeometryObjectD<dim>::shared_from_this;

    virtual GeometryObject::Type getType() const override;

    static constexpr const char* NAME = dim == 2 ?
                ("separator" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("separator" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const override;

    /**
     * Separators typically have no materials, so this just return nullptr.
     * @param p point
     * @return @c nullptr
     */
    virtual shared_ptr<Material> getMaterial(const DVec& p) const override;

    //virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryObject>, Box, DVec>>& dest, const PathHints* path = 0) const override;

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const override;

    virtual void getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path = 0) const override;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* = 0) const override;

/*    inline void getLeafsToVec(std::vector< shared_ptr<const GeometryObject> >& dest) const override {
        dest.push_back(this->shared_from_this());
    }

    inline std::vector< shared_ptr<const GeometryObject> > getLeafs() const override {
        return { this->shared_from_this() };
    }*/

    virtual bool hasInSubtree(const GeometryObject& el) const override;

    virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const override;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool=false) const override;

    virtual std::size_t getChildrenCount() const override { return 0; }

    virtual shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override;

    virtual shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation = 0) const override;

    // void extractToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints* = 0) const {
    //     if (predicate(*this)) dest.push_back(static_pointer_cast< const GeometryObjectD<dim> >(this->shared_from_this()));
    // }

    virtual bool contains(const DVec& p) const override;

    /*virtual bool intersects(const Box& area) const override {
        return this->getBoundingBox().intersects(area); //TODO ?? maybe set area to empty
    }*/

};

extern template struct PLASK_API GeometryObjectSeparator<2>;
extern template struct PLASK_API GeometryObjectSeparator<3>;

/**
 * Gap in one, choose at compile time, direction.
 * @tparam dim number of dimensions
 * @tparam direction direction of gap, from 0 to dim-1
 * @ingroup GEOMETRY_OBJ
 */
template < int dim, int direction >
struct Gap1D: public GeometryObjectSeparator<dim> {

    static_assert(direction < dim, "direction must be from 0 to dim-1");

    typedef typename GeometryObjectSeparator<dim>::DVec DVec;
    typedef typename GeometryObjectSeparator<dim>::Box Box;

    static constexpr const char* NAME = "gap";              ///< name of gap type, used as XML tag name when write to XML
    static constexpr const char* XML_SIZE_ATTR = "size";    ///< name of size attribute in XML

    virtual std::string getTypeName() const { return NAME; }

    /// Size of gap.
    double size;

    /**
     * Construct gap with given size.
     * @param size size of gap
     */
    Gap1D(double size = 0.0): size(size) {}

    virtual Box getBoundingBox() const override {
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

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames&) const override {
        dest_xml_object.attr(XML_SIZE_ATTR, size);
    }

};

}    // namespace plask

#endif // PLASK__GEOMETRY_SEPARATOR_H

