#ifndef PLASK_GUI_MODEL_EXT_MAP_IMPL_LEAF_H
#define PLASK_GUI_MODEL_EXT_MAP_IMPL_LEAF_H

/** @file
 * This file includes implementation of geometry elements model extensions for leafs. Do not include it directly (see map.h).
 */

#include <plask/geometry/leaf.h>

template <int dim>
QString printBlock(const plask::Block<dim>& toPrint) {
    return QString(QObject::tr("block%1d\nsize: %2"))
            .arg(dim).arg(QString(boost::lexical_cast<std::string>(toPrint.size).c_str()));
}


#endif // PLASK_GUI_MODEL_EXT_MAP_IMPL_LEAF_H
