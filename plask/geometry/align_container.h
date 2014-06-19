#ifndef PLASK__GEOMETRY_ALIGN_CONTAINER_H
#define PLASK__GEOMETRY_ALIGN_CONTAINER_H

/** @file
This file contains containers of geometries objects which align all children in one direction and allow to explicitly choose coordinates in other directions.
*/

#include "container.h"
#include "align.h"
#include "../utils/metaprog.h"
#include <utility>

namespace plask {

template <int dim, typename Primitive<dim>::Direction alignDirection> 
using AlignContainerChildAligner = typename chooseType<dim-2,
        align::Aligner<direction3D(DirectionWithout<3, direction3D(alignDirection)>::value2D)>,
        align::Aligner<DirectionWithout<3, direction3D(alignDirection)>::valueLower, DirectionWithout<3, direction3D(alignDirection)>::valueHigher>
    >::type;

/**
 * Containers of geometries objects which align all children in one direction (typically to top/left/center)
 * and allow to explicitly choose coordinates in other directions.
 * @ingroup GEOMETRY_OBJ
 */
template <int dim, typename Primitive<dim>::Direction alignDirection>
struct AlignContainer: public WithAligners<GeometryObjectContainer<dim>, AlignContainerChildAligner<dim, alignDirection>> {

    static constexpr const char* NAME = dim == 2 ?
                ("align" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("align" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    typedef align::Aligner<direction3D(alignDirection)> Aligner;

    typedef AlignContainerChildAligner<dim, alignDirection> ChildAligner;

    static ChildAligner defaultAligner();

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
    Aligner aligner;

    /**
     * Create and set up translation object for new child.
     * @param el
     * @param aligner
     * @return
     */
    shared_ptr<TranslationT> newTranslation(const shared_ptr<ChildType>& el, ChildAligner aligner);

protected:
    virtual shared_ptr<GeometryObject> changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const;

public:

    AlignContainer(const Aligner& aligner)
        : aligner(aligner)
    {}

    /// Called by child.change signal, update heights call this change
    void onChildChanged(const GeometryObject::Event& evt) {
        if (evt.isResize()) {
            auto child = const_cast<TranslationT&>(evt.source<TranslationT>());
            auto chAligner = this->getAlignerFor(child);
            if (!chAligner.isNull())
                align::align(child, chAligner, aligner);
        }
        GeometryObjectContainer<dim>::onChildChanged(evt);
    }

    /**
     * Get aligner which is use to align object in alignDirection.
     * @return aligner which is use to align object
     */
    const Aligner& getAligner() const {
        return aligner;
    }

    /**
     * Set aligner which will be used to align object in alignDirection.
     * @param new_aligner new aligner to use
     */
    void setAligner(const Aligner& new_aligner) {
        aligner = new_aligner;
    }

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param aligner
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(shared_ptr<ChildType> el, ChildAligner aligner = defaultAligner());

    /**
     * Add new child (trasnlated) to end of children vector.
     * @param el new child
     * @param aligner
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, ChildAligner aligner = defaultAligner()) {
        this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, aligner);
    }

    virtual std::string getTypeName() const {
        return NAME;
    }

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;    //this attributes

    //void writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const;   //child attributes

};

template <> AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::ChildAligner AlignContainer<2, Primitive<2>::DIRECTION_TRAN>::defaultAligner();
template <> AlignContainer<2, Primitive<2>::DIRECTION_VERT>::ChildAligner AlignContainer<2, Primitive<2>::DIRECTION_VERT>::defaultAligner();
template <> AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::ChildAligner AlignContainer<3, Primitive<3>::DIRECTION_TRAN>::defaultAligner();
template <> AlignContainer<3, Primitive<3>::DIRECTION_VERT>::ChildAligner AlignContainer<3, Primitive<3>::DIRECTION_VERT>::defaultAligner();
template <> AlignContainer<3, Primitive<3>::DIRECTION_LONG>::ChildAligner AlignContainer<3, Primitive<3>::DIRECTION_LONG>::defaultAligner();

PLASK_API_EXTERN_TEMPLATE_STRUCT(AlignContainer<2, Primitive<2>::DIRECTION_TRAN>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(AlignContainer<2, Primitive<2>::DIRECTION_VERT>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(AlignContainer<3, Primitive<3>::DIRECTION_LONG>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(AlignContainer<3, Primitive<3>::DIRECTION_TRAN>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(AlignContainer<3, Primitive<3>::DIRECTION_VERT>)

}   // namespace plask

#endif //  PLASK__GEOMETRY_ALIGN_CONTAINER_H
