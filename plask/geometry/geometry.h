#ifndef PLASK__GEOMETRY_H
#define PLASK__GEOMETRY_H

/** @file
This file includes all plask geometry headers.
*/

/**
@page geometry Geometry
@section geometry_about About
Geometry in PLaSK is represented by directed, acyclic graph which have geometry elements in vertexes (see plask::GeometryElement).
Very often this graph is tree, in which each element has pointers to his children.
Terminology (names) from trees theory are used in PLaSK (element C is child of P only if there is arc from P to C, and then P is called parent of C).

Each vertex is represent by object of class inherited from plask::GeometryElementType.
This objects must be hold by plask::shared_ptr (which allow to automatically delete it in proper time).

Types of geometry elements / vertexes (see also plask::GeometryElementType):
- leafs (terminal nodes) - each doesn't have children and stores information about material;
- transformation nodes - each has exactly one child, and represent this child after transformation (sometimes transformation element is in different space than it's child);
- containers - each has one or more children and represents figure which consist of this children.

Each geometry element (vertex) has local coordinate system (in 2d or 3d space).
Systems of all geometries elements which use Cartesian systems have common direction of axis.
All 2d elements lies in tran-up (or R-Z) plane.

@section geometry_paths Paths
You can add each geometry element object to graph more than one time.
If you do this, you will sometimes need to use paths to point concrete appearance of element (which is more than once) in graph.

Each path is represented by object of plask::PathHints class.

plask::PathHints consist of, zero or more, hints (objects of plask::PathHints::Hint class).
Each hint is a pair which show arc (on path) from container to one child of this container.
Hints are returned by methods which adds new elements to containers, and can be added to plask::PathHints by a += operator:
@code
PathHints mypath;
//...
mypath += container_element.add(child_element);
container_element.add(child_element);
container_element.add(child_element);
//Now child_element is three times in container_element.
//In mypath is arc from container_element to first appearance of child_element.
//...
//Remove one (pointed by mypath - first) appearance of child_element from container_element:
container_element.remove(mypath);
//Still child_element is in container_element two times.
//Remove all child_elements from container_element:
container_element.remove(child_element);
@endcode

*/

#include "leaf.h"
#include "transform.h"
#include "container.h"

#include "manager.h"

namespace plask {

}       // namespace plask

#endif
