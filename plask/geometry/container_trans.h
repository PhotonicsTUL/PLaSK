#ifndef PLASK__GEOMETRY_CONTAINER_TRANS_H
#define PLASK__GEOMETRY_CONTAINER_TRANS_H

#include "container.h"

namespace plask {

/**
 * Geometry objects container in which every child has an associated translation vector.
 * @ingroup GEOMETRY_OBJ
 */
//TODO some implementation are naive, and can be done faster with some caches
template < int dim >
struct TranslationContainer: public GeometryObjectContainer<dim> {

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::Box Box;

    /// Type of this child.
    typedef typename GeometryObjectContainer<dim>::ChildType ChildType;

    /// Type of translation geometry object in space of this.
    typedef typename GeometryObjectContainer<dim>::TranslationT TranslationT;

    using GeometryObjectContainer<dim>::children;
    using GeometryObjectContainer<dim>::shared_from_this;

    static constexpr const char* NAME = dim == 2 ?
                ("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const { return NAME; }

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        shared_ptr<TranslationT> trans_geom(new TranslationT(el, translation));
        this->connectOnChildChanged(*trans_geom);
        children.push_back(trans_geom);
        this->fireChildrenInserted(children.size()-1, children.size());
        return PathHints::Hint(shared_from_this(), trans_geom);
    }

    /**
     * Add new child (trasnlated) to end of children vector.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, translation);
    }

    virtual void writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const;

protected:
    virtual shared_ptr<GeometryObject> changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const;

};

} // namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_TRANS_H
