#ifndef PLASK__GEOMETRY_LEAF_H
#define PLASK__GEOMETRY_LEAF_H

/** @file
This file contains geometry objects leafs classes.
*/

#include "object.hpp"
#include "reader.hpp"
#include "../manager.hpp"
#include "../material/db.hpp"

namespace plask {

class GeometryReader;

/**
 * Template of base classes for all leaf nodes.
 * @tparam dim number of dimensions
 * @ingroup GEOMETRY_OBJ
 */
template <int dim> struct PLASK_API GeometryObjectLeaf : public GeometryObjectD<dim> {
    typedef typename GeometryObjectD<dim>::DVec DVec;
    typedef typename GeometryObjectD<dim>::Box Box;
    using GeometryObjectD<dim>::getBoundingBox;
    using GeometryObjectD<dim>::shared_from_this;

  public:
    struct PLASK_API MaterialProvider {
        virtual shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const = 0;

        /**
         * Get material only if it this provider represents solid material (if getMaterial returns value independent
         * from arguments).
         * @return material or nullptr if it is not solid
         */
        virtual shared_ptr<Material> singleMaterial() const = 0;

        virtual MaterialProvider* clone() const = 0;

        /**
         * Get representative material of this provider (typically material which is returned in center of object).
         * @return representative material of this provider
         */
        virtual shared_ptr<Material> getRepresentativeMaterial() const = 0;

        virtual XMLWriter::Element& writeXML(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const = 0;

        virtual bool isUniform(Primitive<3>::Direction direction) const = 0;

        virtual ~MaterialProvider() {}
    };

    struct PLASK_API SolidMaterial : public MaterialProvider {
        shared_ptr<Material> material;

        SolidMaterial() = default;

        SolidMaterial(shared_ptr<Material> material) : material(material) {}

        virtual shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& PLASK_UNUSED(thisObj),
                                                 const DVec& PLASK_UNUSED(p)) const override {
            return material;
        }

        shared_ptr<Material> singleMaterial() const override { return material; }

        SolidMaterial* clone() const override { return new SolidMaterial(material); }

        shared_ptr<Material> getRepresentativeMaterial() const override { return material; }

        bool isUniform(Primitive<3>::Direction PLASK_UNUSED(direction)) const override { return true; }

        XMLWriter::Element& writeXML(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
    };

    struct PLASK_API GradientMaterial : public MaterialProvider {
        shared_ptr<MaterialsDB::MixedCompositionFactory> materialFactory;

        GradientMaterial(shared_ptr<MaterialsDB::MixedCompositionFactory> materialFactory)
            : materialFactory(materialFactory) {}

        shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const override {
            Box b = thisObj.getBoundingBox();  // TODO sth. faster. we only need vert() coordinates, we can also cache
                                               // lower and height
            return (*materialFactory)((p.vert() - b.lower.vert()) / b.height());
        }

        shared_ptr<Material> singleMaterial() const override { return shared_ptr<Material>(); }

        GradientMaterial* clone() const override { return new GradientMaterial(materialFactory); }

        shared_ptr<Material> getRepresentativeMaterial() const override { return (*materialFactory)(0.5); }

        bool isUniform(Primitive<3>::Direction direction) const override {
            return direction != Primitive<3>::DIRECTION_VERT;
        }

        XMLWriter::Element& writeXML(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
    };

    struct PLASK_API DraftGradientMaterial : public GradientMaterial {

        DraftGradientMaterial(shared_ptr<MaterialsDB::MixedCompositionFactory> materialFactory)
            : GradientMaterial(materialFactory) {}

        shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const override {
            return (*this->materialFactory)(0.5);
        }

        bool isUniform(Primitive<3>::Direction direction) const override { return true; }
    };

  protected:
    std::unique_ptr<MaterialProvider> materialProvider;

  public:
    GeometryReader& readMaterial(GeometryReader& src);

    // shared_ptr<Material> material;  //TODO support for XML (checking if MixedCompositionFactory is solid), add
    // singleMaterial

    /**
     * Construct leaf with uninitialized material (all material getting methods returns nullptr).
     */
    GeometryObjectLeaf<dim>() : materialProvider(new SolidMaterial()) {}

    /**
     * Copy-constructor
     */
    GeometryObjectLeaf<dim>(const GeometryObjectLeaf<dim>& src) : materialProvider(src.materialProvider->clone()) {}

    /**
     * Construct leaf which uses solid material.
     * @param material to set (solid on all surface)
     */
    GeometryObjectLeaf<dim>(shared_ptr<Material> material) : materialProvider(new SolidMaterial(material)) {}

    /**
     * Construct leaf which uses linearly changeble material.
     * @param materialTopBottom materials to set (lineary changeble), first is the material on top of this, the second
     * is on bottom of this
     */
    GeometryObjectLeaf<dim>(shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
        : materialProvider(new GradientMaterial(materialTopBottom)) {}

    /**
     * Get representative material of this leaf (typically material which is returned in center of object).
     * @return representative material of this
     */
    shared_ptr<Material> getRepresentativeMaterial() const { return materialProvider->getRepresentativeMaterial(); }

    /**
     * Get material only if it this leaf is solid (has assign exactly one material).
     * @return material or nullptr if it is not solid
     */
    shared_ptr<Material> singleMaterial() const { return materialProvider->singleMaterial(); }

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
    void setMaterialFast(shared_ptr<Material> new_material) { materialProvider.reset(new SolidMaterial(new_material)); }

    /**
     * Set new graded material. Do not inform listeners about the change.
     * @param materialTopBottom materials to set (lineary changeble), first is the material on top of this, the second
     * is on bottom of this
     */
    void setMaterialTopBottomCompositionFast(shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom) {
        materialProvider.reset(new GradientMaterial(materialTopBottom));
    }

    /**
     * Set new material material.
     * @param materialTopBottom materials to set (lineary changeble), first is the material on top of this, the second
     * is on bottom of this
     */
    void setMaterialTopBottomComposition(shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom) {
        setMaterialTopBottomCompositionFast(materialTopBottom);
        this->fireChanged();
    }

    /**
     * Set new draft graded material.
     * @param materialTopBottom materials to set (lineary changeble), first is the material on top of this, the second
     * is on bottom of this
     */
    void setMaterialDraftTopBottomCompositionFast(shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom) {
        materialProvider.reset(new DraftGradientMaterial(materialTopBottom));
    }

    /**
     * Set new custom material provider. Do not inform listeners about the change.
     * \param provider custom material provider to set; this class takes ownership of this provider
     */
    void setMaterialProviderFast(MaterialProvider* provider) { materialProvider.reset(provider); }

    /**
     * Set new custom material provider.
     * \param provider custom material provider to set; this class takes ownership of this provider
     */
    void setMaterialProvider(MaterialProvider* provider) {
        setMaterialProviderFast(provider);
        this->fireChanged();
    }

    /**
     * Return pointer to the material provider
     */
    const MaterialProvider* getMaterialProvider() const { return materialProvider.get(); }

    GeometryObject::Type getType() const override;

    shared_ptr<Material> getMaterial(const DVec& p) const override;

    void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                               std::vector<Box>& dest,
                               const PathHints* path = 0) const override;

    void getObjectsToVec(const GeometryObject::Predicate& predicate,
                         std::vector<shared_ptr<const GeometryObject>>& dest,
                         const PathHints* path = 0) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate,
                           std::vector<DVec>& dest,
                           const PathHints* = 0) const override;

    bool hasInSubtree(const GeometryObject& el) const override;

    GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const override;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool = false) const override;

    std::size_t getChildrenCount() const override { return 0; }

    shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override;

    shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer,
                                                    Vec<3, double>* translation = 0) const override;

    shared_ptr<GeometryObject> deepCopy(
        std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const override;

    // void extractToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObjectD<dim>
    // > >& dest, const PathHints* = 0) const {
    //     if (predicate(*this)) dest.push_back(static_pointer_cast< const GeometryObjectD<dim>
    //     >(this->shared_from_this()));
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
template <int dim> struct PLASK_API Block : public GeometryObjectLeaf<dim> {
    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::Box Box;

    static const char* NAME;

    std::string getTypeName() const override;

    /**
     * Size and upper corner of block. Lower corner is a zero vector.
     */
    DVec size;

    /**
     * Set size and inform observers about changes.
     * @param new_size new size to set
     */
    void setSize(DVec&& new_size) {
        for (int i = 0; i != dim; ++i)
            if (new_size[i] < 0.) new_size[i] = 0.;
        size = new_size;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set size and inform observers about changes.
     * @param vecCtrArg new size to set (parameters to Vec<dim> constructor)
     */
    template <typename... VecCtrArg> void setSize(VecCtrArg&&... vecCtrArg) {
        this->setSize(DVec(std::forward<VecCtrArg>(vecCtrArg)...));
    }

    /**
     * Create block.
     * @param size size/upper corner of block
     * @param material block material
     */
    explicit Block(const DVec& size = Primitive<dim>::ZERO_VEC,
                   const shared_ptr<Material>& material = shared_ptr<Material>())
        : GeometryObjectLeaf<dim>(material), size(size) {
        for (int i = 0; i != dim; ++i)
            if (size[i] < 0.) this->size[i] = 0.;
    }

    explicit Block(const DVec& size, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
        : GeometryObjectLeaf<dim>(materialTopBottom), size(size) {}

    explicit Block(const Block& src) : GeometryObjectLeaf<dim>(src), size(src.size) {}

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    // bool intersects(const Box& area) const { return this->getBoundingBox().intersects(area); }

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Block>(*this); }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

template <> void Block<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;
template <> void Block<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

template <>
void Block<2>::addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                                    unsigned max_steps,
                                    double min_step_size) const;
template <>
void Block<3>::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                    unsigned max_steps,
                                    double min_step_size) const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(Block<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Block<3>)

/**
 * Construct Block with the same dimenstion as bouding box of @p to_change.
 * @param material material of the constructed Block
 * @param to_change geometry object
 * @param translation[out] set to position (lower corner) of @c to_change bouding box
 * @param draft should draft graded material be created?
 * @return Block<to_change->getDimensionsCount()> object
 */
PLASK_API shared_ptr<GeometryObject> changeToBlock(const SolidOrGradientMaterial& material,
                                                   const shared_ptr<const GeometryObject>& to_change,
                                                   Vec<3, double>& translation, bool draft = false);


namespace details {

    // Read alternative attributes
    inline static double readAlternativeAttrs(GeometryReader& reader, const std::string& attr1, const std::string& attr2) {
        auto value1 = reader.source.getAttribute<double>(attr1);
        auto value2 = reader.source.getAttribute<double>(attr2);
        if (value1) {
            if (value2) throw XMLConflictingAttributesException(reader.source, attr1, attr2);
            if (*value1 < 0.) throw XMLBadAttrException(reader.source, attr1, boost::lexical_cast<std::string>(*value1));
            return *value1;
        } else {
            if (!value2) {
                if (reader.manager.draft)
                    return 0.0;
                else
                    throw XMLNoAttrException(reader.source, format("{0}' or '{1}", attr1, attr2));
            }
            if (*value2 < 0.) throw XMLBadAttrException(reader.source, attr2, boost::lexical_cast<std::string>(*value2));
            return *value2;
        }
    }

    template <typename BlockType> inline static void setupBlock2D3D(GeometryReader& reader, BlockType& block) {
        block.size.tran() = readAlternativeAttrs(reader, "d" + reader.getAxisTranName(), "width");
        block.size.vert() = readAlternativeAttrs(reader, "d" + reader.getAxisVertName(), "height");
        block.readMaterial(reader);
        reader.source.requireTagEnd();
    }
}

typedef Block<2> Rectangle;
typedef Block<3> Cuboid;

}  // namespace plask

#endif  // PLASK__GEOMETRY_LEAF_H
