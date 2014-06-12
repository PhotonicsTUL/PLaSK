#ifndef PLASK__GEOMETRY_LEAF_H
#define PLASK__GEOMETRY_LEAF_H

/** @file
This file contains geometry objects leafs classes.
*/

#include "object.h"
#include "../material/db.h"

namespace plask {

class GeometryReader;

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

protected:

    struct PLASK_API MaterialProvider {
        virtual shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const = 0;

        /**
         * Get material only if it this provider represents solid material (if getMaterial returns value independent from arguments).
         * @return material or nullptr if it is not solid
         */
        virtual shared_ptr<Material> singleMaterial() const = 0;

        virtual MaterialProvider* clone() const = 0;

        /**
         * Get representative material of this provider (typically material which is returned in center of object).
         * @return representative material of this provider
         */
        virtual shared_ptr<Material> getRepresentativeMaterial() const = 0;

        virtual XMLWriter::Element& writeXML(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const = 0;

        virtual bool singleMaterialInBB(Primitive<3>::Direction direction) const = 0;

        virtual ~MaterialProvider() {}
    };

    struct PLASK_API SolidMaterial: public MaterialProvider {
        shared_ptr<Material> material;

        SolidMaterial() = default;

        SolidMaterial(shared_ptr<Material> material): material(material) {}

        virtual shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const {
            return material;
        }

        virtual shared_ptr<Material> singleMaterial() const {
            return material;
        }

        SolidMaterial* clone() const {
            return new SolidMaterial(material);
        }

        virtual shared_ptr<Material> getRepresentativeMaterial() const {
            return material;
        }

        virtual bool singleMaterialInBB(Primitive<3>::Direction direction) const { return true; }

        virtual XMLWriter::Element& writeXML(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const;
    };

    struct PLASK_API MixedCompositionMaterial: public MaterialProvider {

        shared_ptr<MaterialsDB::MixedCompositionFactory> materialFactory;

        MixedCompositionMaterial(shared_ptr<MaterialsDB::MixedCompositionFactory> materialFactory): materialFactory(materialFactory) {}

        virtual shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const {
            Box b = thisObj.getBoundingBox(); //TODO sth. faster. we only need vert() coordinates, we can also cache lower and height
            return (*materialFactory)((p.vert() - b.lower.vert()) / b.height());
        }

        virtual shared_ptr<Material> singleMaterial() const {
            return shared_ptr<Material>();
        }

        MixedCompositionMaterial* clone() const {
            return new MixedCompositionMaterial(materialFactory);
        }

        virtual shared_ptr<Material> getRepresentativeMaterial() const {
            return (*materialFactory)(0.5);
        }

        virtual bool singleMaterialInBB(Primitive<3>::Direction direction) const { return direction != Primitive<3>::DIRECTION_VERT; }

        virtual XMLWriter::Element& writeXML(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const;
    };

    std::unique_ptr<MaterialProvider> materialProvider;

public:

    GeometryReader & readMaterial(GeometryReader &src);

    //shared_ptr<Material> material;  //TODO support for XML (checking if MixedCompositionFactory is solid), add singleMaterial

    /**
     * Construct leaf with uninitialized material (all material getting methods returns nullptr).
     */
    GeometryObjectLeaf<dim>(): materialProvider(new SolidMaterial()) {}

    /**
     * Construct leaf which uses solid material.
     * @param material to set (solid on all surface)
     */
    GeometryObjectLeaf<dim>(shared_ptr<Material> material): materialProvider(new SolidMaterial(material)) {}

    /**
     * Construct leaf which uses lineary changeble material.
     * @param materialTopBottom materials to set (lineary changeble), first is the material on top of this, the second is on bottom of this
     */
    GeometryObjectLeaf<dim>(shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom): materialProvider(new MixedCompositionMaterial(materialTopBottom)) {}

    bool singleMaterialInBB(Primitive<3>::Direction direction) const override {
        return materialProvider->singleMaterialInBB(direction);
    }

    /**
     * Get representative material of this leaf (typically material which is returned in center of object).
     * @return representative material of this
     */
    shared_ptr<Material> getRepresentativeMaterial() const {
        return materialProvider->getRepresentativeMaterial();
    }

    /**
     * Get material only if it this leaf is solid (has assign exactly one material).
     * @return material or nullptr if it is not solid
     */
    shared_ptr<Material> singleMaterial() const {
        return materialProvider->singleMaterial();
    }

    /**
     * Set new, solid on all surface, material.
     * @param new_material material to set
     */
    void setMaterial(shared_ptr<Material> new_material) {
        materialProvider.reset(new SolidMaterial(new_material));
        this->fireChanged();
    }

    /**
     * Set new, solid on all surface, material. Do not inform listeners about the change.
     * @param new_material material to set
     */
    void setMaterialFast(shared_ptr<Material> new_material) {
        materialProvider.reset(new SolidMaterial(new_material));
    }

    /**
     * Set new, lineary changeble, material.
     * @param materialTopBottom materials to set (lineary changeble), first is the material on top of this, the second is on bottom of this
     */
    void setMaterialTopBottomCompositionFast(shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom) {
        materialProvider.reset(new MixedCompositionMaterial(materialTopBottom));
    }

    /**
     * Set new, lineary changeble, material. Do not inform listeners about the change.
     * @param materialTopBottom materials to set (lineary changeble), first is the material on top of this, the second is on bottom of this
     */
    void setMaterialTopBottomComposition(shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom) {
        setMaterialTopBottomCompositionFast(materialTopBottom);
        this->fireChanged();
    }

    virtual GeometryObject::Type getType() const { return GeometryObject::TYPE_LEAF; }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return this->contains(p) ? materialProvider->getMaterial(*this, p) : shared_ptr<Material>();
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

    virtual bool hasInSubtree(const GeometryObject& el) const {
        return &el == this;
    }

    virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const {
        return GeometryObject::Subtree( &el == this ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
    }

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool=false) const {
        return GeometryObject::Subtree( this->contains(point) ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
    }

    virtual std::size_t getChildrenCount() const { return 0; }

    virtual shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const {
        throw OutOfBoundsException("GeometryObjectLeaf::getChildNo", "child_no");
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

#ifndef PLASK_EXPORTS
extern template struct PLASK_API GeometryObjectLeaf<2>;
extern template struct PLASK_API GeometryObjectLeaf<3>;
#endif

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
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
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

    explicit Block(const DVec& size, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom): GeometryObjectLeaf<dim>(materialTopBottom), size(size) {}

    virtual Box getBoundingBox() const {
        return Box(Primitive<dim>::ZERO_VEC, size);
    }

    virtual bool contains(const DVec& p) const {
        return this->getBoundingBox().contains(p);
    }

    virtual bool intersects(const Box& area) const {
        return this->getBoundingBox().intersects(area);
    }

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

};

template <> void Block<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;
template <> void Block<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

#ifndef PLASK_EXPORTS
extern template struct PLASK_API Block<2>;
extern template struct PLASK_API Block<3>;
#endif

PLASK_API shared_ptr<GeometryObject> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryObject>& to_change, Vec<3, double>& translation);

typedef Block<2> Rectangle;
typedef Block<3> Cuboid;

}    // namespace plask

#endif // PLASK__GEOMETRY_LEAF_H
