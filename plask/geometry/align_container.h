#ifndef PLASK__GEOMETRY_ALIGN_CONTAINER_H
#define PLASK__GEOMETRY_ALIGN_CONTAINER_H

/** @file
This file includes containers of geometries objects which align all children in one direction and allow to explicitly choose coordinates in other directions.
*/

#include "container.h"
#include "align.h"
#include "../utils/metaprog.h"

namespace plask {

/**
 * Containers of geometries objects which align all children in one direction (typically to top/left/center)
 * and allow to explicitly choose coordinates in other directions.
 */
//TODO implementation
template <int dim, typename Primitive<dim>::Direction alignDirection>
class AlignContainer: public GeometryObjectContainer<dim> {

    typedef typename chooseType<dim-2, align::OneDirectionAligner<direction2Dto3D(alignDirection)>, align::OneDirectionAligner<alignDirection> >::type Aligner;

    /**
     * Aligner which is use to align object in alignDirection.
     */
    std::unique_ptr<Aligner> aligner;

public:

    AlignContainer(const Aligner& aligner)
        : aligner(aligner.cloneUnique())
    {}

    /// Called by child.change signal, update heights call this change
    void onChildChanged(const GeometryObject::Event& evt) {
        if (evt.isResize()) aligner->align(evt.source());
        GeometryObjectContainer<dim>::onChildChanged(evt);
    }

    const Aligner& getAligner() const {
        return *aligner;
    }

    void setAligner(const Aligner& new_aligner) const {
        aligner = new_aligner.cloneUnique();
    }

};

}   // namespace plask

#endif //  PLASK__GEOMETRY_ALIGN_CONTAINER_H
