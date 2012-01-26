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

@section geometry_XML Geometry representation in XML format
Geometry can be read from XML content (for details about reading XML see @ref geometry_create).

Example of XML which describe geometry:
@verbatim
<geometry axis="x, y">
    <container2d name="trans_cont">
        <block2d x="5" y="3" name="block_5_3" material="exampleMaterial"/>
        <child x="3" y="3">
            <ref name="block_5_3"/>
        </child>
    </container2d>
    <stack2d repeat="4" name="mystack">
        <ref name="block_5_3"/>
        <child x="-5">
            <ref name="c1"/>
        </child>
        <block2d x="8" y="12" material="exampleMaterial"/>
    </stack2d>
</geometry>
@endverbatim

Above XML describe geometry in 2d space, which includes two containers.

First container (described by @a container2d tag), has name "trans_cont" (see name attribute) and has 2 (identical) children.
Its child is rectangle block (@a block2d tag) with name "block_5_3" and size 5x3 (see x and y attributes).
First, it occur at point (0, 0) (which is default), and second, it occur at point (3, 3) (which is given in child tag attributes).
Second appearance of block in container is given by ref tag. This tag represent reference to early defined element and it require only one attribute: name of element which was early defined.

Second container is describe by tag stack2d and has name "mystack". Because it has repeat attribute it will be represent by instantiation of plask::MultiStackContainer. This container includes:
early defined elements with names "block_5_3" and "c1" (second one is translated inside "mystack" in x direction by -5), and rectangle block with size 8x12.

Geometry tag has axis attribute which define names of axises. Each other tag inside geometry also can have axis attribute.
In such cases this attribute define local (valid up to end of this tag with axis attribute) names of axis.

@section geometry_create How to create and use geometry graph? (examples)

Geometry graph can be created:
- manually (by construct its element), for example we construct container same as first container in example from section @ref geometry_XML:
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
- from XML content, by using plask::GeometryManager, for example (we suppose that example_file_name.xml file includes content showed in @ref geometry_XML section):
@code
plask::GeometryManager geometry;
//read XML content from file:
geometry.loadFromFile("example_file_name.xml");
//use:
plask::shared_ptr< plask::Block<2> > block_5_3 = geometry.requireElement< plask::Block<2> >("block_5_3");
assert(block_5_3->getBoundingBox().lower == plask::vec(0.0, 0.0));
assert(block_5_3->getBoundingBox().upper == plask::vec(5.0, 3.0));
assert(block_5_3->getMaterial(plask::vec(4.0, 2.0)) == exampleMaterial);
assert(block_5_3->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
plask::shared_ptr< plask::TranslationContainer<2> > container = geometry.requireElement< plask::TranslationContainer<2> >("trans_cont");
assert(container->getBoundingBox() == plask::Rect2d(plask::vec(0.0, 0.0), plask::vec(8.0, 6.0)));
assert(container->getMaterial(plask::vec(6.0, 6.0)) == exampleMaterial);
assert(container->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
@endcode

@section geometry_paths Paths
You can add each geometry element object to graph more than one time.
If you do this, you will sometimes need to use paths to point concrete appearance of element (which is more than once) in graph.

Each path is represented by object of plask::PathHints class.

plask::PathHints consist of, zero or more, hints (objects of plask::PathHints::Hint class).
Each hint is a pair which show arc (on path) from container to one child of this container.
Hints are returned by methods which adds new elements to containers, and can be added to plask::PathHints by a += operator:
@code
plask::PathHints mypath;
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

@section geometry_newelements Writing new geometry elements

To write new geometry element you should:
-# write a class which directly or indirectly inherit from plask::GeometryElement and implement all its abstract methods,
-# write a reader function which allow to read your element from XML (see plask::GeometryReader and plask::GeometryReader::element_read_f for details),
-# register your reader in global registry, creating global instance of plask::GeometryReader::RegisterElementReader class.

Good base classes for geometries elements are, for example:
- plask::GeometryElementLeaf instantiations - for leaf elements,
- plask::GeometryElementTransform or plask::GeometryElementChangeSpace instantiations - for transformation elements,
- plask::GeometryElementContainer instantiations - for containers,
- plask::GeometryElementD instantiations - generic.
*/

#include "leaf.h"
#include "transform.h"
#include "container.h"

#include "manager.h"

namespace plask {

}       // namespace plask

#endif
