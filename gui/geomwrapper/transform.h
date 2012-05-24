#ifndef GUI_GEOMETRY_WRAPPER_TRANSFORM_H
#define GUI_GEOMETRY_WRAPPER_TRANSFORM_H

/** @file
 * This file includes implementation of geometry elements model extensions for transforms. Do not include it directly (see register.h).
 */

#include "element.h"

#include <plask/geometry/transform.h>

template <int dim>
struct TranslationWrapper: public ElementWrapperFor< plask::Translation<dim> > {

    virtual QString toStr() const;

    virtual void draw(QPainter& painter) const;

};

#endif //GUI_GEOMETRY_WRAPPER_TRANSFORM_H
