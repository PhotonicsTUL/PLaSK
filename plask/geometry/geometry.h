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

@section geometry_create How to create and use geometry graph? (examples)

Geometry graph can be created:
- manually (by construct its element), for example:
@code
//create 2d solid block (with size 5x3) filled with example material:
plask::shared_ptr< plask::Block<2> > block_5_3(new plask::Block<2>(plask::vec(5.0, 3.0), exampleMaterial));
//check some block_5_3 properties:
assert(block_5_3->getBoundingBox().lower == plask::vec(0.0, 0.0));
assert(block_5_3->getBoundingBox().upper == plask::vec(5.0, 3.0));
assert(block_5_3->getMaterial(plask::vec(4.0, 2.0)) == exampleMaterial);
assert(block_5_3->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
//create 2d container and add two children (blocks) to it:
plask::shared_ptr<plask::TranslationContainer<2>> container(new plask::TranslationContainer<2>);
container->add(block_5_3);
container->add(block_5_3, plask::vec(3.0, 3.0));
//now our graphs has 3 vertexes: 1 container and 2 (identical) blocks in it
//check some container properties:
assert(container->getBoundingBox() == plask::Rect2d(plask::vec(0.0, 0.0), plask::vec(8.0, 6.0)));
assert(container->getMaterial(plask::vec(6.0, 6.0)) == exampleMaterial);
assert(container->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
@endcode
- from XML content (for example, read from file), by using plask::GeometryManager, for example:
@code
plask::GeometryManager geometry;
geometry.loadFromFile("example_file_name.xml");
TODO
@endcode

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
