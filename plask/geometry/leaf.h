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
struct PLASK_API GeometryObjectLeaf: public GeometryObjectD<dim> {

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

        virtual bool isUniform(Primitive<3>::Direction direction) const = 0;

        virtual ~MaterialProvider() {}
    };

    struct PLASK_API SolidMaterial: public MaterialProvider {
        shared_ptr<Material> material;

        SolidMaterial() = default;

        SolidMaterial(shared_ptr<Material> material): material(material) {}

        virtual shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const override {
            return material;
        }

        virtual shared_ptr<Material> singleMaterial() const override {
            return material;
        }

        SolidMaterial* clone() const override {
            return new SolidMaterial(material);
        }

        virtual shared_ptr<Material> getRepresentativeMaterial() const override {
            return material;
        }

        virtual bool isUniform(Primitive<3>::Direction direction) const override { return true; }

        virtual XMLWriter::Element& writeXML(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const override;
    };

    struct PLASK_API MixedCompositionMaterial: public MaterialProvider {

        shared_ptr<MaterialsDB::MixedCompositionFactory> materialFactory;

        MixedCompositionMaterial(shared_ptr<MaterialsDB::MixedCompositionFactory> materialFactory): materialFactory(materialFactory) {}

        virtual shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const override {
            Box b = thisObj.getBoundingBox(); //TODO sth. faster. we only need vert() coordinates, we can also cache lower and height
            return (*materialFactory)((p.vert() - b.lower.vert()) / b.height());
        }

        virtual shared_ptr<Material> singleMaterial() const override {
            return shared_ptr<Material>();
        }

        MixedCompositionMaterial* clone() const override {
            return new MixedCompositionMaterial(materialFactory);
        }

        virtual shared_ptr<Material> getRepresentativeMaterial() const override {
            return (*materialFactory)(0.5);
        }

        virtual bool isUniform(Primitive<3>::Direction direction) const override { return direction != Primitive<3>::DIRECTION_VERT; }

        virtual XMLWriter::Element& writeXML(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const override;
    };

    std::unique_ptr<MaterialProvider> materialProvider;

public:

    GeometryReader& readMaterial(GeometryReader &src);

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

    bool isUniform(Primitive<3>::Direction direction) const override;

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

    virtual GeometryObject::Type getType() const override;

    virtual shared_ptr<Material> getMaterial(const DVec& p) const override;

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const override;

    virtual void getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path = 0) const override;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* = 0) const override;

    virtual bool hasInSubtree(const GeometryObject& el) const override;

    virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const override;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool=false) const override;

    virtual std::size_t getChildrenCount() const override { return 0; }

    virtual shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override;

    virtual shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation = 0) const override;

    // void extractToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints* = 0) const {
    //     if (predicate(*this)) dest.push_back(static_pointer_cast< const GeometryObjectD<dim> >(this->shared_from_this()));
    // }

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(GeometryObjectLeaf<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(GeometryObjectLeaf<3>)

/**
Represent figure which, depends from @p dim is:
- for dim = 2 - rectangle,
- for dim = 3 - cuboid.
@tparam dim number of dimensions
@ingroup GEOMETRY_OBJ
*/
template <int dim>
struct PLASK_API Block: public GeometryObjectLeaf<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::Box Box;

    static const char* NAME;

    virtual std::string getTypeName() const override;

    /**
     * Size and upper corner of block. Lower corner is zeroed vector.
     */
    DVec size;

    /**
     * Set size and inform observers about changes.
     * @param new_size new size to set
     */
    void setSize(DVec&& new_size) {
        for(int i = 0; i != dim; ++i)
            if (new_size[i] < 0.) new_size[i] = 0.;
        size = new_size;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set size and inform observers about changes.
     * @param vecCtrArg new size to set (parameters to Vec<dim> constructor)
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
        : GeometryObjectLeaf<dim>(material), size(size) {
        for(int i = 0; i != dim; ++i)
            if (size[i] < 0.) this->size[i] = 0.;
    }

    explicit Block(const DVec& size, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom): GeometryObjectLeaf<dim>(materialTopBottom), size(size) {}

    virtual Box getBoundingBox() const override;

    virtual bool contains(const DVec& p) const override;

    bool intersects(const Box &area) const {
        return this->getBoundingBox().intersects(area);
    }

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

};

template <> void Block<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;
template <> void Block<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(Block<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Block<3>)

/**
 * Construct Block with the same dimenstion as bouding box of @p to_change.
 * @param material material og the constructed Block
 * @param to_change geometry object
 * @param translation[out] set to position (lower corner) of @c to_change bouding box
 * @return Block<to_change->getDimensionsCount()> object
 */
PLASK_API shared_ptr<GeometryObject> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryObject>& to_change, Vec<3, double>& translation);

typedef Block<2> Rectangle;
typedef Block<3> Cuboid;

}    // namespace plask

#endif // PLASK__GEOMETRY_LEAF_H
