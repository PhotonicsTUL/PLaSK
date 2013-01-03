#ifndef PLASK__GEOMETRY_ALIGN_CONTAINER_H
#define PLASK__GEOMETRY_ALIGN_CONTAINER_H

/** @file
This file includes containers of geometries objects which align all children in one direction and allow to explicitly choose coordinates in rest directions.
*/

#include "align.h"
#include "../utils/metaprog.h"

namespace plask {

/**
 * Containers of geometries objects which align all children in one direction (typically to top/left/center)
 * and allow to explicitly choose coordinates in rest directions.
 */
//TODO implementation
template <int dim, typename Primitive<dim>::Direction alignDirection>
class AlignContainer {

    typedef typename chooseType<dim-2, align::OneDirectionAligner<direction2Dto3D(alignDirection)>, align::OneDirectionAligner<alignDirection> >::type Aligner;

    /**
     * Aligner which is use to align object in alignDirection.
     */
    std::unique_ptr<Aligner> aligner;
    
};

}   // namespace plask

#endif //  PLASK__GEOMETRY_ALIGN_CONTAINER_H
