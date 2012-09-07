#ifndef PLASK__GEOMETRY_LEAF_H
#define PLASK__GEOMETRY_LEAF_H

/** @file
This file includes geometry elements leafs classes.
*/

#include "element.h"

namespace plask {

/**
 * Template of base classes for all leaf nodes.
 * @tparam dim number of dimensions
 */
template < int dim >
struct GeometryElementLeaf: public GeometryElementD<dim> {

    typedef typename GeometryElementD<dim>::DVec DVec;
    typedef typename GeometryElementD<dim>::Box Box;
    using GeometryElementD<dim>::getBoundingBox;
    using GeometryElementD<dim>::shared_from_this;

    shared_ptr<Material> material;

    GeometryElementLeaf<dim>(shared_ptr<Material> material): material(material) {}

    void setMaterial(shared_ptr<Material> new_material) {
        material = new_material;
        this->fireChanged();
    }

    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_LEAF; }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return this->includes(p) ? material : shared_ptr<Material>();
    }

    virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryElement>, Box, DVec>>& dest, const PathHints* path = 0) const {
        dest.push_back( std::tuple<shared_ptr<const GeometryElement>, Box, DVec>(this->shared_from_this(), this->getBoundingBox(), Primitive<dim>::ZERO_VEC) );
    }

    virtual void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const {
        if (predicate(*this))
            dest.push_back(this->getBoundingBox());
    }

    inline std::vector<Box> getLeafsBoundingBoxes() const {
        return { this->getBoundingBox() };
    }

    inline std::vector<Box> getLeafsBoundingBoxes(const PathHints&) const {
        return { this->getBoundingBox() };
    }

    virtual void getElementsToVec(const GeometryElement::Predicate& predicate, std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints* path = 0) const {
        if (predicate(*this)) dest.push_back(this->shared_from_this());
    }

    virtual void getPositionsToVec(const GeometryElement::Predicate& predicate, std::vector<DVec>& dest, const PathHints* = 0) const {
        if (predicate(*this)) dest.push_back(Primitive<dim>::ZERO_VEC);
    }

    inline void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest) const {
        dest.push_back(this->shared_from_this());
    }

    inline std::vector< shared_ptr<const GeometryElement> > getLeafs() const {
        return { this->shared_from_this() };
    }

    /*virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        return { std::make_pair(shared_from_this(), Primitive<dim>::ZERO_VEC) };
    }*/

    virtual bool isInSubtree(const GeometryElement& el) const {
        return &el == this;
    }

    virtual GeometryElement::Subtree getPathsTo(const GeometryElement& el, const PathHints* path = 0) const {
        return GeometryElement::Subtree( &el == this ? this->shared_from_this() : shared_ptr<const GeometryElement>() );
    }

    virtual GeometryElement::Subtree getPathsTo(const DVec& point) const {
        return GeometryElement::Subtree( this->includes(point) ? this->shared_from_this() : shared_ptr<const GeometryElement>() );
    }

    virtual std::size_t getChildrenCount() const { return 0; }

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const {
        throw OutOfBoundException("GeometryElementLeaf::getChildAt", "child_nr");
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<const GeometryElement> result(this->shared_from_this());
        changer.apply(result, translation);
        return result;
    }

};

/**
Represent figure which, depends from @p dim is:
- for dim = 2 - rectangle,
- for dim = 3 - cuboid.

Block is filled with one material.
@tparam dim number of dimensions
*/
template <int dim>
struct Block: public GeometryElementLeaf<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementLeaf<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementLeaf<dim>::Box Box;
    
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
        this->fireChanged(GeometryElement::Event::RESIZE);
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
        : GeometryElementLeaf<dim>(material), size(size) {}

    virtual Box getBoundingBox() const {
        return Box(Primitive<dim>::ZERO_VEC, size);
    }

    virtual bool includes(const DVec& p) const {
        return this->getBoundingBox().includes(p);
    }

    virtual bool intersects(const Box& area) const {
        return this->getBoundingBox().intersects(area);
    }
    
    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_element, const AxisNames& axes) const;

};

shared_ptr<GeometryElement> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryElement>& to_change, Vec<3, double>& translation);

typedef Block<2> Rectangle;
typedef Block<3> Cuboid;

}    // namespace plask

#endif // PLASK__GEOMETRY_LEAF_H
