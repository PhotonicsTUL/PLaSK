#include "geometry.h"

#include <plask/geometry/transform.h>
#include <plask/geometry/mirror.h>
#include <plask/geometry/clip.h>
#include <plask/geometry/intersection.h>

namespace plask { namespace python {

extern AxisNames current_axes;

template <int dim>
static bool Transfrom__contains__(const GeometryObjectTransform<dim>& self, shared_ptr<typename GeometryObjectTransform<dim>::ChildType> child) {
    if (self.getChild() == child) return true;
    return false;
}


/// Initialize class GeometryObjectTransform for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectTransform, "GeometryObjectTransform", "Base class for all "," geometry transforms.") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectTransform, GeometryObjectD<dim>)
        .add_property("item",
                      (shared_ptr<typename GeometryObjectTransform<dim>::ChildType> (GeometryObjectTransform<dim>::*)()) &GeometryObjectTransform<dim>::getChild,
                      &GeometryObjectTransform<dim>::setChild, "Transformed item.")
        .def("__contains__", &Transfrom__contains__<dim>)
    ;
}


// Some helpful wrappers
template <int dim>
shared_ptr<Translation<dim>> Translation_constructor1(shared_ptr<GeometryObjectD<dim>> object, const Vec<dim,double>& trans) {
    return make_shared<Translation<dim>>(object, trans);
}

template <int dim> struct Translation_constructor2 {};
template <> struct Translation_constructor2<2> {
    static inline shared_ptr<Translation<2>> call(shared_ptr<GeometryObjectD<2>> object, double c0, double c1) {
        return make_shared<Translation<2>>(object, Vec<2,double>(c0, c1));
    }
    const static py::detail::keywords<3> args;
};
const py::detail::keywords<3> Translation_constructor2<2>::args = (py::arg("item"), py::arg("c0"), py::arg("c1"));
template <> struct Translation_constructor2<3> {
    static inline shared_ptr<Translation<3>> call(shared_ptr<GeometryObjectD<3>> object, double c0, double c1, double c2) {
        return make_shared<Translation<3>>(object, Vec<3,double>(c0, c1, c2));
    }
    const static py::detail::keywords<4> args;
};
const py::detail::keywords<4> Translation_constructor2<3>::args = (py::arg("item"), py::arg("c0"), py::arg("c1"), py::arg("c2"));


std::string GeometryObject__repr__(const shared_ptr<GeometryObject>& self);

template <int dim>
static std::string Translation__str__(const Translation<dim>& self) {
    std::stringstream out;
    out << "(";
    try {
        std::string str = py::extract<std::string>(py::object(self.getChild()).attr("__repr__")());
        out << str;
    } catch (py::error_already_set) {
        PyErr_Clear();
        out << GeometryObject__repr__(self.getChild());
    }
    out << ", plask.vec("; for (int i = 0; i < dim; ++i) out << pyformat(self.translation[i]) << (i!=dim-1 ? "," : ")");
    out << ")";
    return out.str();
}

template <int dim>
static std::string Translation__repr__(const Translation<dim>& self) {
    return format("plask.geometry.Translation%1%D%2%", dim, Translation__str__<dim>(self));
}

DECLARE_GEOMETRY_ELEMENT_23D(Translation, "Translation",
    "Transform that holds a translated geometry object together with its translation\n"
    "vector ("," version).\n\n"
    "Args:\n"
    "item (GeomeryObject): Item to translate.\n"
    "translation (plask.vec): Translation vector.\n"
    "cx (float): Component of the translation vector along the *x* axis.\n")
{
    GEOMETRY_ELEMENT_23D(Translation, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Translation_constructor1<dim>, py::default_call_policies(), (py::arg("item"), py::arg("translation"))))
    .def("__init__", py::make_constructor(&Translation_constructor2<dim>::call, py::default_call_policies(), Translation_constructor2<dim>::args))
    .def_readwrite("translation", &Translation<dim>::translation, "Translation vector.")
    .def("__str__", &Translation__str__<dim>)
    .def("__repr__", &Translation__repr__<dim>)
    ;
}


template <typename Cls>
shared_ptr<Cls> Mirror_constructor1(size_t axis, shared_ptr<typename Cls::ChildType> child) {
    if (axis >= Cls::DIM) throw ValueError("Wrong axis number.");
    return make_shared<Cls>(typename Primitive<Cls::DIM>::Direction(axis), child);
}

template <typename Cls>
shared_ptr<Cls> Mirror_constructor2(const std::string& axis, shared_ptr<typename Cls::ChildType> child) {
    size_t no = current_axes[axis] + Cls::DIM - 3;
    return make_shared<Cls>(typename Primitive<Cls::DIM>::Direction(no), child);
}

template <typename Cls>
std::string getFlipDir(const Cls& self) { return current_axes[self.flipDir]; }

template <typename Cls>
void setFlipDir(Cls& self, py::object val) {
    try {
        size_t no = current_axes[py::extract<std::string>(val)] + Cls::DIM - 3;
        self.flipDir = typename Primitive<Cls::DIM>::Direction(no);
    } catch (py::error_already_set) {
        PyErr_Clear();
        size_t no = py::extract<size_t>(val);
        if (no >= Cls::DIM) throw ValueError("Wrong axis number.");
        self.flipDir = typename Primitive<Cls::DIM>::Direction(no);
    }
}

template <typename Cls> int getFlipDirNr(const Cls& self) { return int(self.flipDir); }


DECLARE_GEOMETRY_ELEMENT_23D(Flip, "Flip",
    "Transfer that flips the geometry object along a specified axis\n"
    "("," version).\n\n"
    "Args:\n"
    "   axis (float or str): Flip axis number or name.\n"
    "   item (GeometryObject): Geometry object to flip.\n\n"
    "The effect of this transform is a mirror reflection of a specified geometry\n"
    "object. Its local coordinate system is negated along the specified axis.\n\n"
    "The difference between the flip and a mirror is that the flip replaces the\n"
    "original object with its flipped version.\n\n"
    "See also:\n"
    "   :class:`plask.geometry.Mirror`")
{
    GEOMETRY_ELEMENT_23D(Flip, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Mirror_constructor1<Flip<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .def("__init__", py::make_constructor(&Mirror_constructor2<Flip<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .add_property("axis", &getFlipDir<Flip<dim>>, &setFlipDir<Flip<dim>>, "Flip axis.")
    .add_property("axis_nr", &getFlipDirNr<Flip<dim>>, &setFlipDir<Flip<dim>>, "Number of the flip axis.")
    ;
}

DECLARE_GEOMETRY_ELEMENT_23D(Mirror, "Mirror",
    "Transfer that mirrors the geometry object along the specified  axis\n"
    "("," version).\n\n"
    "Args:\n"
    "   axis (float or str): Flip axis number or name.\n"
    "   item (GeometryObject): Geometry object to flip.\n\n"
    "The effect of this transform is an original object with an added mirror\n"
    "reflection. The mirroring is done with respect to the axis, so the whole\n"
    "object should be bouned within one half-plane of its local coordinate\n"
    "system.\n\n"
    "The difference between the mirror and a flip is that the flip replaces the\n"
    "original object with its flipped version.\n\n"
    "See also:\n"
    "   :class:`plask.geometry.Flip`")
{
    GEOMETRY_ELEMENT_23D(Mirror, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Mirror_constructor1<Mirror<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .def("__init__", py::make_constructor(&Mirror_constructor2<Mirror<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .add_property("axis", &getFlipDir<Mirror<dim>>, &setFlipDir<Mirror<dim>>, "Mirror axis.")
    .add_property("axis_nr", &getFlipDirNr<Mirror<dim>>, &setFlipDir<Mirror<dim>>, "Number of the mirror axis.")
    ;
}

template <int dim>
shared_ptr<Clip<dim>> Clip_constructor1(shared_ptr<GeometryObjectD<dim>> object, const typename Primitive<dim>::Box& clip_box) {
    return make_shared<Clip<dim>>(object, clip_box);
}

template <int dim> struct Clip_constructor2 {};
template <> struct Clip_constructor2<2> {
    static inline shared_ptr<Clip<2>> call(shared_ptr<GeometryObjectD<2>> object, double left, double bottom, double right, double top) {
            return make_shared<Clip<2>>(object, Box2D(left, bottom, right, top));
    }
    const static py::detail::keywords<5> args;
};
const py::detail::keywords<5> Clip_constructor2<2>::args = (py::arg("item"),
                                                            py::arg("left")=-INFINITY, py::arg("bottom")=-INFINITY,
                                                            py::arg("right")=INFINITY, py::arg("top")=INFINITY);

template <> struct Clip_constructor2<3> {
    static inline shared_ptr<Clip<3>> call(shared_ptr<GeometryObjectD<3>> object, double back, double left, double bottom, double front, double right, double top) {
            return make_shared<Clip<3>>(object, Box3D(back, left, bottom, front, right, top));
    }
    const static py::detail::keywords<7> args;
};
const py::detail::keywords<7> Clip_constructor2<3>::args = (py::arg("item"),
                                                            py::arg("back")=-INFINITY, py::arg("left")=-INFINITY, py::arg("bottom")=-INFINITY,
                                                            py::arg("front")=INFINITY, py::arg("right")=INFINITY, py::arg("top")=INFINITY);


template <int dim> inline const char* ClipName();
template <> inline const char* ClipName<2>() { return "Clip2D"; }
template <> inline const char* ClipName<3>() { return "Clip3D"; }

template <int dim> inline const char* ClipDoc();
template <> inline const char* ClipDoc<2>() { return
    "Clip2D(item, box)\n"
    "Clip2D(item, left, bottom, right, top)\n\n"
    "Transform that clips the held geometry object to the specified clip-box\n"
    "(2D version).\n\n"
    "This transform is used to limit the size of any complicated geometry object to\n"
    "the specified rectangular box. This way, you can easily change e.g. a circle\n"
    "to a half- or quarter-circle. In order to use this transform, you must\n"
    "explicitly specify the coordinates of the clip-box in the local coordinates of\n"
    "the clipped object. However, the original object is never expanded, co you can\n"
    "freely make the box very large, or even infinite (which means no clipping at\n"
    "this side).\n\n"
    "Args:\n"
    "    item (GeometryObject2D): Object to clip.\n"
    "    left (float): Left side of the clipping box. *inf* by default.\n"
    "    bottom (float): Bottom side of the clipping box. *inf* by default.\n"
    "    right (float): Right side of the clipping box. *inf* by default.\n"
    "    top (float): Top side of the clipping box. *inf* by default.\n"
    "    box (Box2D): Clipping box.\n\n"
    "Example:\n"
    "    To make a half-circle with the flat bottom:\n\n"
    "    >>> circle = plask.geometry.Circle(2, 'GaAs')\n"
    "    >>> half_circle = plask.geometry.Clip2D(circle, bottom=0)\n"
    ;
}
template <> inline const char* ClipDoc<3>() { return
    "Clip3D(item, box)\n"
    "Clip3D(item, back, left, bottom, front, right, top)\n\n"
    "Transform that clips the held geometry object to the specified clip-box\n"
    "(3D version).\n\n"
    "This transform is used to limit the size of any complicated geometry object to\n"
    "the specified rectangular box. This way, you can easily change e.g. a cylinder\n"
    "to a half- or quarter-cilinder. In order to use this transform, you must\n"
    "explicitly specify the coordinates of the clip-box in the local coordinates of\n"
    "the clipped object. However, the original object is never expanded, co you can\n"
    "freely make the box very large, or even infinite (which means no clipping at\n"
    "this side).\n\n"
    "Args:\n"
    "    item (GeometryObject3D): Object to clip.\n"
    "    back (float): Back side of the clipping box. *inf* by default.\n"
    "    left (float): Left side of the clipping box. *inf* by default.\n"
    "    bottom (float): Bottom side of the clipping box. *inf* by default.\n"
    "    front (float): Front side of the clipping box. *inf* by default.\n"
    "    right (float): Right side of the clipping box. *inf* by default.\n"
    "    top (float): Top side of the clipping box. *inf* by default.\n"
    "    box (Box3D): Clipping box.\n\n"
    "Example:\n"
    "    To make a half-cylinder with the flat front side:\n\n"
    "    >>> cylinder = plask.geometry.Cylinder(2, 1, 'GaAs')\n"
    "    >>> half_cylinder = plask.geometry.Clip3D(cylinder, front=0)\n"
    ;
}

template <int dim>
inline static void init_Clip()
{
    py::class_<Clip<dim>, shared_ptr<Clip<dim>>, py::bases<GeometryObjectTransform<dim>>, boost::noncopyable>(
    ClipName<dim>(), ClipDoc<dim>(), py::no_init)
        .def("__init__", py::make_constructor(&Clip_constructor1<dim>, py::default_call_policies(), (py::arg("item"), py::arg("box"))))
        .def("__init__", py::make_constructor(&Clip_constructor2<dim>::call, py::default_call_policies(), Clip_constructor2<dim>::args))
        .def_readwrite("clip_box", &Clip<dim>::clipBox, "Clipping box.")
        ;
}

template <int dim>
shared_ptr<Intersection<dim>> Intersection_constructor(shared_ptr<GeometryObjectD<dim>> object, shared_ptr<GeometryObjectD<dim>> shape) {
    return make_shared<Intersection<dim>>(object, shape);
}

template <int dim> inline const char* IntersectionName();
template <> inline const char* IntersectionName<2>() { return "Intersection2D"; }
template <> inline const char* IntersectionName<3>() { return "Intersection3D"; }

template <int dim> inline const char* IntersectionDoc();
template <> inline const char* IntersectionDoc<2>() { return
    "Intersection2D(item, shape)\n\n"
    "Transform that clips the held geometry object to the specified envelope\n"
    "(2D version).\n\n"
    "This transform is a more advanced version of :class:`~plask.geometry.Clip2D`.\n"
    "Instead of a simple box, you can specify any geometry object as a clipping\n"
    "envelope. The material of the evelope is ignored.\n\n"
    "Args:\n"
    "    item (GeometryObject2D): Object to clip.\n"
    "    shape (GeometryObject2D): Object to serve as a clipping envelope.\n\n"
//TODO
//    "Example:\n"
//    "    To make a half-circle with the flat bottom:\n\n"
//    "    >>> circle = plask.geometry.Circle(2, 'GaAs')\n"
//    "    >>> half_circle = plask.geometry.Intersection2D(circle, bottom=0)\n"
    ;
}
template <> inline const char* IntersectionDoc<3>() { return
    "Intersection3D(item, shape)\n\n"
    "Transform that clips the held geometry object to the specified envelope\n"
    "(3D version).\n\n"
    "This transform is a more advanced version of :class:`~plask.geometry.Clip3D`.\n"
    "Instead of a simple box, you can specify any geometry object as a clipping\n"
    "envelope. The material of the evelope is ignored.\n\n"
    "Args:\n"
    "    item (GeometryObject3D): Object to clip.\n"
    "    shape (GeometryObject3D): Object to serve as a clipping envelope.\n\n"
//TODO
//    "Example:\n"
//    "    To make a half-cylinder with the flat front side:\n\n"
//    "    >>> cylinder = plask.geometry.Cylinder(2, 1, 'GaAs')\n"
//    "    >>> half_cylinder = plask.geometry.Intersection3D(cylinder, front=0)\n"
    ;
}

template <int dim>
inline static void init_Intersection()
{
    py::class_<Intersection<dim>, shared_ptr<Intersection<dim>>, py::bases<GeometryObjectTransform<dim>>, boost::noncopyable>(
    IntersectionName<dim>(), IntersectionDoc<dim>(), py::no_init)
        .def("__init__", py::make_constructor(&Intersection_constructor<dim>, py::default_call_policies(), (py::arg("item")=shared_ptr<GeometryObjectD<dim>>(), py::arg("shape")=shared_ptr<GeometryObjectD<dim>>())))
        .add_property("envelope", &Intersection<dim>::getEnvelope, &Intersection<dim>::setEnvelope,
                      "Clipping envelope.\n\n"
                      "This is a geometry object that serves as a clipping envelope. The main item\n"
                      "of this transform is clipped to the shape of the envelope.")
    ;
}

void register_geometry_changespace();

void register_geometry_transform()
{
    init_GeometryObjectTransform<2>();
    init_GeometryObjectTransform<3>();

    register_geometry_changespace();

    init_Translation<2>();
    init_Translation<3>();

    init_Flip<2>();
    init_Flip<3>();

    init_Mirror<2>();
    init_Mirror<3>();

    init_Clip<2>();
    init_Clip<3>();

    init_Intersection<2>();
    init_Intersection<3>();
}

}} // namespace plask::python
