#ifndef PLASK__GEOMETRY_H
#define PLASK__GEOMETRY_H

/**
@section geometry Geometry
@subsection geometry_about About
Geometry in PLaSK is represented by directed, acyclic graph which have geometry elements in vertices (see plask::GeometryElement).
Very often this graph is tree, in which each element has pointers to his children.
Terminology (names) from trees theory are used in PLaSK (element C is child of P only if there is arc from P to C, and then P is called parent of C).

Types of geometry elements / vertices (see also @plask::GeometryElementType):
- leafs (terminal nodes) - each doesn't have children and stores information about material
- transformation nodes - each has exactly one child, and represent this child after transformation (sometimes transformation element is in different space than it's child)
- containers - each has one or more children

*/

#include "element.h"

namespace plask {

}       // namespace plask

#endif
