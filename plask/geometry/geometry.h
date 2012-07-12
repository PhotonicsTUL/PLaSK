#ifndef PLASK__GEOMETRY_H
#define PLASK__GEOMETRY_H

/** @file
This file includes all plask geometry headers.
*/

/**
@page geometry Geometry
@section geometry_about About
Geometry in PLaSK is represented by a directed, acyclic graph, which has geometry elements in its vertices
(see plask::GeometryElement). Very often this graph is tree, in which each element has pointers to its children.
PLaSK uses terminology inhrtited from the tree theory i.e. element C is a child of P only if there is arc from P to C,
and then P is called the parent of C).

Each vertex is represented by an object of class inherited from the plask::GeometryElement type.
Each such object must be hold by plask::shared_ptr (which automatically deletes it in the proper time).

There are following types of geometry elements / vertices (see also plask::GeometryElementType):
- \b leafs (terminal nodes) : don't have children and store information about material;
- \b transformation nodes : each has exactly one child, and represents this child after transformation (sometimes transformation element is in different space than it's child);
- \b containers :  each has one or more children and represents a substructure which consist of these children (e.g. a stack).

Each geometry element (vertex) has local coordinate system (in 2D or 3D space).
Systems of all geometry elements which use Cartesian coordinates have common orientation of axes.
All 2D elements lie in the tran-up (or R-Z) plane.

@section geometry_XML Geometry representation in XML format
Geometry can be read from an XML content (for details about reading XML see @ref geometry_create).

Example of XML which describes geometry:
\code{.xml}
<geometry axes="xy">
    <cartesian2d name="cartesian geometry" length="2">
        <container2d name="trans_cont">
            <rectangle x="5" y="3" name="block_5_3" material="exampleMaterial"/>
            <child x="3" y="3">
                <ref name="block_5_3"/>
            </child>
        </container2d>
    </cartesian2d>
    <cylindrical name="cylindrical geometry">
        <stack2d repeat="4" name="mystack">
            <ref name="block_5_3"/>
            <child x="-5">
                <ref name="c1"/>
            </child>
            <rectangle x="8" y="12" material="exampleMaterial"/>
        </stack2D>
     </cylindrical>
</geometry>
\endcode

Above XML describes 2 geometries, each is in 2D space, and each includes one container.
First geometry is cartesian, has name "cartesian geometry", and length equal to 2.
Second geometry is cylindrical and has name "cylindrical geometry".

Container in first geometry (described by the \c container2D tag), has name "trans_cont" (see \c name attribute) and has 2 (identical) children.
They are actually the same rectangular block (\c rectangle tag) with name "block_5_3" and size 5x3 (see \c x and \c y attributes).
Its first instance is locates at point (0, 0) (which is a default), and the second one is located at point (3, 3) (which is given in the \c child tag attributes).
Second appearance of the block in container is given by the \c ref tag.
This tag represents reference to an ealier defined element and it requires only one attribute,
namely the name of the element which was earlier defined.

Container in second geometry is described by a tag \c stack2D and is named "mystack". Because it has a \c repeat attribute it will be
represented by an object of class plask::MultiStackContainer. This container has two children:
earlier defined elements with names "block_5_3" and "c1" (the second one is translated inside "mystack" in x direction by -5),
and a block with size 8x12.

Geometry tag has an \c axes attribute which defines names of the axes. Each other tag inside geometry also can have
an \c axes attribute. In such cases this attribute defines local names of axes (i.e. valid up to end of this tag).

@section geometry_create How to create and use geometry graph? (examples)

Geometry graph can be created:
- manually in Python (by constructing its element), for example we construct the same container as the one in the example from section @ref geometry_XML :
\code{.py}
# create 2D solid block (with size 5x3) filled with example material
# (which is an object of class plask.material.Material or its subclass):
block_5_3 = plask.geometry.Rectangle(5.0, 3.0, exampleMaterial)
# check some block_5_3 properties:
assert block_5_3.boundingBox.lower == plask.vec(0.0, 0.0)
assert block_5_3.boundingBox.upper == plask.vec(5.0, 3.0)
assert block_5_3.getMaterial(4.0, 2.0) == exampleMaterial
assert block_5_3.getMaterial(6.0, 2.0) == None
# create 2D container and add two children (blocks) to it:
container = plask.geometry.TranslationContainer2D();
container.append(block_5_3)
container.append(block_5_3, plask.vec(3.0, 3.0))
# now our graphs has 3 vertexes: 1 container and 2 (identical) blocks in it
# check some container properties:
assert container.boundingBox == plask.geometry.Box2D(0.0, 0.0, 8.0, 6.0)
assert container.getMaterial(6.0, 6.0) == exampleMaterial
assert container.getMaterial(6.0, 2.0) == None
\endcode
- manually from C++. The same example as the Python code above:
@code
// create 2D solid block (with size 5x3) filled with example material:
plask::shared_ptr< plask::Rectangle > block_5_3(new plask::Rectangle(plask::vec(5.0, 3.0), exampleMaterial));
// check some block_5_3 properties:
assert(block_5_3->getBoundingBox().lower == plask::vec(0.0, 0.0));
assert(block_5_3->getBoundingBox().upper == plask::vec(5.0, 3.0));
assert(block_5_3->getMaterial(plask::vec(4.0, 2.0)) == exampleMaterial);
assert(block_5_3->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
// create 2D container and add two children (blocks) to it:
plask::shared_ptr<plask::TranslationContainer<2>> container(new plask::TranslationContainer<2>);
container->add(block_5_3);
container->add(block_5_3, plask::vec(3.0, 3.0));
// now our graphs has 3 vertexes: 1 container and 2 (identical) blocks in it
// check some container properties:
assert(container->getBoundingBox() == plask::(plask::vec(0.0, 0.0), plask::vec(8.0, 6.0)));
assert(container->getMaterial(plask::vec(6.0, 6.0)) == exampleMaterial);
assert(container->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
@endcode
- from XML file in Python using Geometry class, for example  (we suppose that example_file_name.xml file includes content showed in @ref geometry_XML section):
\code{.py}
geometry = plask.geometry.Geometry("example_file_name.xml")
# use:
block_5_3 = geometry.element("block_5_3")
assert block_5_3.boundingBox.lower == plask.vec(0.0, 0.0)
assert block_5_3.boundingBox.upper == plask.vec(5.0, 3.0)
assert block_5_3.getMaterial(plask.vec(4.0, 2.0)) == exampleMaterial
assert block_5_3.getMaterial(plask.vec(6.0, 2.0)) == None
container = geometry.element("trans_cont")
assert container.boundingBox == plask.geometry.Box2D(plask.vec(0.0, 0.0), plask.vec(8.0, 6.0))
assert container.getMaterial(plask.vec(6.0, 6.0)) == exampleMaterial
assert container.getMaterial(plask.vec(6.0, 2.0)) == None
\endcode
- from XML content in C++, by using plask::Manager, for example:
@code
plask::Manager geometry;
// read XML content from file:
geometry.loadFromFile("example_file_name.xml");
// use:
plask::shared_ptr< plask::Rectangle > block_5_3 = geometry.requireElement< plask::Rectangle >("block_5_3");
assert(block_5_3->getBoundingBox().lower == plask::vec(0.0, 0.0));
assert(block_5_3->getBoundingBox().upper == plask::vec(5.0, 3.0));
assert(block_5_3->getMaterial(plask::vec(4.0, 2.0)) == exampleMaterial);
assert(block_5_3->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
plask::shared_ptr< plask::TranslationContainer<2> > container = geometry.requireElement< plask::TranslationContainer<2> >("trans_cont");
assert(container->getBoundingBox() == plask::(plask::vec(0.0, 0.0), plask::vec(8.0, 6.0)));
assert(container->getMaterial(plask::vec(6.0, 6.0)) == exampleMaterial);
assert(container->getMaterial(plask::vec(6.0, 2.0)) == nullptr);
@endcode

@section geometry_paths Paths
You can add each geometry element object to graph more than one time.
If you do this, you will sometimes need to use paths to point to particular instance of this element,
which can appear more than once in the graph.

Each path is represented by an object of plask::PathHints class.

plask::PathHints consist of zero or more hints (objects of plask::PathHints::Hint class).
Each hint is a pair which represent a connection between the container and one of its children.
Hints are returned by methods which adds new elements to containers, and can be added to plask::PathHints by a += operator:
@code
plask::PathHints mypath;
// ...
mypath += container_element.push_back(child_element);
container_element.push_back(child_element);
container_element.push_back(child_element);
// Now child_element is three times in container_element.
// There is an arc from container_element to first appearance of child_element in mypath.
// ...
// Remove one (pointed by mypath i.e. the first) appearance of child_element from container_element:
container_element.remove(mypath);
// There are still two instances child_element is in container_element.
// Remove all child_elements from container_element:
container_element.remove(child_element);
@endcode

In Python this can be done in similar way (see User Manual for more examples):

\code{.py}
mypath += container_element.append(child_element)
container_element.append(child_element)
container_element.append(child_element)
# Now child_element is three times in container_element.
# There is an arc from container_element to first appearance of child_element in mypath
# ...
# Remove one (pointed by mypath i.e. the first) appearance of child_element from container_element:
del container_element[mypath]
# There are still two instances child_element is in container_element.
# Remove all child_elements from container_element:
del container_element[child_element]
\endcode

@section geometry_newelements Writing new geometry elements

To write new geometry element you should:
-# write a class which directly or indirectly inherit from plask::GeometryElement and implement all its abstract methods,
-# write a reader function which allow to read your element from XML (see plask::GeometryReader and plask::GeometryReader::element_read_f for details),
-# register your reader in global registry, creating global instance of plask::GeometryReader::RegisterElementReader class.
-# write Python binding to your class using boost::python::class_ construction. This class should be registered in geometry scope.

Good base classes for geometries elements are, for example:
- plask::GeometryElementLeaf instantiations - for leaf elements,
- plask::GeometryElementTransform or plask::GeometryElementTransformSpace instantiations - for transformation elements,
- plask::GeometryElementContainer instantiations - for containers,
- plask::GeometryElementD instantiations - generic.
*/

#include "leaf.h"
#include "transform.h"
#include "container.h"

namespace plask {

}       // namespace plask

#endif
