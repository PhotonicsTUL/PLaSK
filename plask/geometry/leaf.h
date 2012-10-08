#ifndef PLASK__GEOMETRY_LEAF_H
#define PLASK__GEOMETRY_LEAF_H

/** @file
This file includes geometry objects leafs classes.
*/

#include "object.h"

namespace plask {

/**
 * Template of base classes for all leaf nodes.
 * @tparam dim number of dimensions
 * @ingroup GEOMETRY_OBJ
 */
template < int dim >
struct GeometryObjectLeaf: public GeometryObjectD<dim> {

    typedef typename GeometryObjectD<dim>::DVec DVec;
    typedef typename GeometryObjectD<dim>::Box Box;
    using GeometryObjectD<dim>::getBoundingBox;
    using GeometryObjectD<dim>::shared_from_this;

    shared_ptr<Material> material;

    GeometryObjectLeaf<dim>(shared_ptr<Material> material): material(material) {}

    void setMaterial(shared_ptr<Material> new_material) {
        material = new_material;
        this->fireChanged();
    }

    virtual GeometryObject::Type getType() const { return GeometryObject::TYPE_LEAF; }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return this->includes(p) ? material : shared_ptr<Material>();
    }

    virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryObject>, Box, DVec>>& dest, const PathHints* path = 0) const {
        dest.push_back( std::tuple<shared_ptr<const GeometryObject>, Box, DVec>(this->shared_from_this(), this->getBoundingBox(), Primitive<dim>::ZERO_VEC) );
    }

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const {
        if (predicate(*this))
            dest.push_back(this->getBoundingBox());
    }

    inline std::vector<Box> getLeafsBoundingBoxes() const {
        return { this->getBoundingBox() };
    }

    inline std::vector<Box> getLeafsBoundingBoxes(const PathHints&) const {
        return { this->getBoundingBox() };
    }

    virtual void getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path = 0) const {
        if (predicate(*this)) dest.push_back(this->shared_from_this());
    }

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* = 0) const {
        if (predicate(*this)) dest.push_back(Primitive<dim>::ZERO_VEC);
    }

    inline void getLeafsToVec(std::vector< shared_ptr<const GeometryObject> >& dest) const {
        dest.push_back(this->shared_from_this());
    }

    inline std::vector< shared_ptr<const GeometryObject> > getLeafs() const {
        return { this->shared_from_this() };
    }

    /*virtual std::vector< std::tuple<shared_ptr<const GeometryObject>, DVec> > getLeafsWithTranslations() const {
        return { std::make_pair(shared_from_this(), Primitive<dim>::ZERO_VEC) };
    }*/

    virtual bool isInSubtree(const GeometryObject& el) const {
        return &el == this;
    }

    virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const {
        return GeometryObject::Subtree( &el == this ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
    }

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool) const {
        return GeometryObject::Subtree( this->includes(point) ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
    }

    virtual std::size_t getChildrenCount() const { return 0; }

    virtual shared_ptr<GeometryObject> getChildAt(std::size_t child_nr) const {
        throw OutOfBoundException("GeometryObjectLeaf::getChildAt", "child_nr");
    }

    virtual shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
        changer.apply(result, translation);
        return result;
    }

    // void extractToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints* = 0) const {
    //     if (predicate(*this)) dest.push_back(static_pointer_cast< const GeometryObjectD<dim> >(this->shared_from_this()));
    // }

};

/**
Represent figure which, depends from @p dim is:
- for dim = 2 - rectangle,
- for dim = 3 - cuboid.

Block is filled with one material.
@tparam dim number of dimensions
@ingroup GEOMETRY_OBJ
*/
template <int dim>
struct Block: public GeometryObjectLeaf<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::Box Box;

    static constexpr const char* NAME = dim == 2 ?
                ("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const { return NAME; }

    /**
     * Size and upper corner of block. Lower corner is zeroed vector.
     */
    DVec size;

    /**
     * Set size and inform observers about changes.
     * @param new_size new size to set
     */
    void setSize(DVec&& new_size) {
        size = new_size;
        this->fireChanged(GeometryObject::Event::RESIZE);
    }

    /**
     * Set size and inform observers about changes.
     * @param vecCtrArg new size to set (parameters to vector constructor)
     */
    template <typename ...VecCtrArg>
    void setSize(VecCtrArg&&... vecCtrArg) {
        this->setSize(DVec(std::forward<VecCtrArg>(vecCtrArg)...));
    }

    /**
     * Create block.
     * @param size size/upper corner of block
     * @param material block material
     */
    explicit Block(const DVec& size = Primitive<dim>::ZERO_VEC, const shared_ptr<Material>& material = shared_ptr<Material>())
        : GeometryObjectLeaf<dim>(material), size(size) {}

    virtual Box getBoundingBox() const {
        return Box(Primitive<dim>::ZERO_VEC, size);
    }

    virtual bool includes(const DVec& p) const {
        return this->getBoundingBox().includes(p);
    }

    virtual bool intersects(const Box& area) const {
        return this->getBoundingBox().intersects(area);
    }

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

};

shared_ptr<GeometryObject> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryObject>& to_change, Vec<3, double>& translation);

typedef Block<2> Rectangle;
typedef Block<3> Cuboid;

}    // namespace plask

#endif // PLASK__GEOMETRY_LEAF_H
