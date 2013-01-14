#ifndef PLASK__GEOMETRY_ALIGN_CONTAINER_H
#define PLASK__GEOMETRY_ALIGN_CONTAINER_H

/** @file
This file includes containers of geometries objects which align all children in one direction and allow to explicitly choose coordinates in other directions.
*/

#include "container.h"
#include "align.h"
#include "../utils/metaprog.h"
#include <utility>

namespace plask {

/**
 * Containers of geometries objects which align all children in one direction (typically to top/left/center)
 * and allow to explicitly choose coordinates in other directions.
 * @ingroup GEOMETRY_OBJ
 */
template <int dim, typename Primitive<dim>::Direction alignDirection>
struct AlignContainer: public GeometryObjectContainer<dim> {

    static constexpr const char* NAME = dim == 2 ?
                ("align" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("align" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    typedef align::AxisAligner<direction3D(alignDirection)> Aligner;

    typedef typename chooseType<dim-2, double, std::pair<double, double> >::type Coordinates;

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

private:
    /**
     * Aligner which is use to align object in alignDirection.
     */
    std::unique_ptr<Aligner> aligner;

    /**
     * Create new translation object.
     * @param el
     * @param place
     * @return
     */
    shared_ptr<TranslationT> newTranslation(const shared_ptr<ChildType>& el, const Coordinates& place);

    /**
     * Create and setup new child (translation object).
     * @param el
     * @param place trasnalation of child in all directions but alignDirection
     * @return
     */
    shared_ptr<TranslationT> newChild(const shared_ptr<ChildType>& el, const Coordinates& place);

    /**
     * Create and setup new child (translation object).
     * @param el
     * @param trasnalation of child in all directions, alignDirection coordinate of this direction will be ignored and overwrite by alginer
     * @return
     */
    shared_ptr<TranslationT> newChild(const shared_ptr<ChildType>& el, const Vec<dim, double>& translation);

protected:
    virtual shared_ptr<GeometryObject> changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const;

public:

    AlignContainer(const Aligner& aligner)
        : aligner(aligner.cloneUnique())
    {}

    /// Called by child.change signal, update heights call this change
    void onChildChanged(const GeometryObject::Event& evt) {
        if (evt.isResize()) aligner->align(const_cast<TranslationT&>(evt.source<TranslationT>()));
        GeometryObjectContainer<dim>::onChildChanged(evt);
    }

    /**
     * Get aligner which is use to align object in alignDirection.
     * @return aligner which is use to align object
     */
    const Aligner& getAligner() const {
        return *aligner;
    }

    /**
     * Set aligner which will be used to align object in alignDirection.
     * @param new_aligner new aligner to use
     */
    void setAligner(const Aligner& new_aligner) {
        aligner = new_aligner.cloneUnique();
    }

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param place trasnalation of child in all directions but alignDirection
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(shared_ptr<ChildType> el, const Coordinates& place);

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param translation trasnalation of child in all directions, alignDirection coordinate of this direction will be ignored and overwrite by alginer
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(shared_ptr<ChildType> el, const Vec<dim, double>& translation = Primitive<dim>::ZERO_VEC);

    /**
     * Add new child (trasnlated) to end of children vector.
     * @param el new child
     * @param place trasnalation of child in all directions but alignDirection
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const Coordinates& place) {
        this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, place);
    }

    /**
     * Add new child (trasnlated) to end of children vector.
     * @param el new child
     * @param translation trasnalation of child in all directions, alignDirection coordinate of this direction will be ignored and overwrite by alginer
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const Vec<dim, double>& translation = Primitive<dim>::ZERO_VEC) {
        this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, translation);
    }

    virtual std::string getTypeName() const {
        return NAME;
    }

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;    //this attributes

    void writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const;   //child attributes

};

}   // namespace plask

#endif //  PLASK__GEOMETRY_ALIGN_CONTAINER_H
