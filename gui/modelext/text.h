#ifndef PLASK_GUI_MODEL_EXT_TEXT_H
#define PLASK_GUI_MODEL_EXT_TEXT_H

/** @file
 * This file includes set of functions to convert some plask types to QString text representation.
 */

#include <plask/geometry/element.h>

#include <QString>
#include <boost/lexical_cast.hpp>


QString toStr(plask::GeometryElement::Type type);

template <int dim, typename T>
inline QString toStr(const plask::Vec<dim, T>& v) {
    return QString(boost::lexical_cast<std::string>(v).c_str());
}

#endif // PLASK_GUI_MODEL_EXT_TEXT_H
