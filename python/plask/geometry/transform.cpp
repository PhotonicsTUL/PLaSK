#include "geometry.hpp"

#include "plask/geometry/transform.hpp"
#include "plask/geometry/mirror.hpp"
#include "plask/geometry/clip.hpp"
#include "plask/geometry/intersection.hpp"

namespace plask { namespace python {

extern AxisNames current_axes;

template <int dim>
static bool Transform__contains__(const GeometryObjectTransform<dim>& self, shared_ptr<typename GeometryObjectTransform<dim>::ChildType> child) {
    if (self.getChild() == child) return true;
    return false;
}


/// Initialize class GeometryObjectTransform for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectTransform, "Transform", "Base class for all "," geometry transforms.") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectTransform, GeometryObjectD<dim>)
        .add_property("item",
                      (shared_ptr<typename GeometryObjectTransform<dim>::ChildType> (GeometryObjectTransform<dim>::*)()) &GeometryObjectTransform<dim>::getChild,
                      &GeometryObjectTransform<dim>::setChild, "Transformed item.")
        .def("__contains__", &Transform__contains__<dim>)

    ;
}


// Some helpful wrappers
template <int dim>
shared_ptr<Translation<dim>> Translation_constructor1(shared_ptr<GeometryObjectD<dim>> object, const Vec<dim,double>& trans) {
    return plask::make_shared<Translation<dim>>(object, trans);
}

template <int dim> struct Translation_constructor2 {};
template <> struct Translation_constructor2<2> {
    static inline shared_ptr<Translation<2>> call(shared_ptr<GeometryObjectD<2>> object, double c0, double c1) {
        return plask::make_shared<Translation<2>>(object, Vec<2,double>(c0, c1));
    }
    const static py::detail::keywords<3> args;
};
const py::detail::keywords<3> Translation_constructor2<2>::args = (py::arg("item"), py::arg("c0"), py::arg("c1"));
template <> struct Translation_constructor2<3> {
    static inline shared_ptr<Translation<3>> call(shared_ptr<GeometryObjectD<3>> object, double c0, double c1, double c2) {
        return plask::make_shared<Translation<3>>(object, Vec<3,double>(c0, c1, c2));
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
    } catch (py::error_already_set&) {
        PyErr_Clear();
        out << GeometryObject__repr__(self.getChild());
    }
    out << ", plask.vec("; for (size_t i = 0; i < dim; ++i) out << pyformat(self.translation[i]) << (i!=dim-1 ? "," : ")");
    out << ")";
    return out.str();
}

template <int dim>
static std::string Translation__repr__(const Translation<dim>& self) {
    return format("plask.geometry.Translation{0}D{1}", dim, Translation__str__<dim>(self));
}

DECLARE_GEOMETRY_ELEMENT_23D(Translation, "Translation",
    u8"Transform that holds a translated geometry object together with its translation\n"
    u8"vector ("," version).\n\n"
    u8"Args:\n"
    u8"    item (GeomeryObject): Item to translate.\n"
    u8"    vec (plask.vec): Translation vector.\n"
    u8"    c# (float): Component of the translation vector along the *#* axis.\n")
{
    GEOMETRY_ELEMENT_23D(Translation, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Translation_constructor1<dim>, py::default_call_policies(), (py::arg("item"), py::arg("vec"))))
    .def("__init__", py::make_constructor(&Translation_constructor2<dim>::call, py::default_call_policies(), Translation_constructor2<dim>::args))
    .def_readwrite("vec", &Translation<dim>::translation, "Translation vector.")
    .def("__str__", &Translation__str__<dim>)
    .def("__repr__", &Translation__repr__<dim>)
    ;
}


template <typename Cls>
shared_ptr<Cls> Mirror_constructor1(size_t axis, shared_ptr<typename Cls::ChildType> child) {
    if (axis >= Cls::DIM) throw ValueError("Wrong axis number.");
    return plask::make_shared<Cls>(typename Primitive<Cls::DIM>::Direction(axis), child);
}

template <typename Cls>
shared_ptr<Cls> Mirror_constructor2(const std::string& axis, shared_ptr<typename Cls::ChildType> child) {
    size_t no = current_axes[axis] + Cls::DIM - 3;
    return plask::make_shared<Cls>(typename Primitive<Cls::DIM>::Direction(no), child);
}

template <typename Cls>
std::string getFlipDir(const Cls& self) { return current_axes[self.flipDir]; }

template <typename Cls>
void setFlipDir(Cls& self, py::object val) {
    try {
        size_t no = current_axes[py::extract<std::string>(val)] + Cls::DIM - 3;
        self.flipDir = typename Primitive<Cls::DIM>::Direction(no);
    } catch (py::error_already_set&) {
        PyErr_Clear();
        size_t no = py::extract<size_t>(val);
        if (no >= Cls::DIM) throw ValueError("Wrong axis number.");
        self.flipDir = typename Primitive<Cls::DIM>::Direction(no);
    }
}

template <typename Cls> int getFlipDirNr(const Cls& self) { return int(self.flipDir); }


DECLARE_GEOMETRY_ELEMENT_23D(Flip, "Flip",
    u8"Transfer that flips the geometry object along a specified axis\n"
    u8"(",u8" version).\n\n"
    u8"Args:\n"
    u8"   axis (float or str): Flip axis number or name.\n"
    u8"   item (GeometryObject): Geometry object to flip.\n\n"
    u8"The effect of this transform is a mirror reflection of a specified geometry\n"
    u8"object. Its local coordinate system is negated along the specified axis.\n\n"
    u8"The difference between the flip and a mirror is that the flip replaces the\n"
    u8"original object with its flipped version.\n\n"
    u8"See also:\n"
    u8"   :class:`plask.geometry.Mirror`")
{
    GEOMETRY_ELEMENT_23D(Flip, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Mirror_constructor1<Flip<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .def("__init__", py::make_constructor(&Mirror_constructor2<Flip<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .add_property("axis", &getFlipDir<Flip<dim>>, &setFlipDir<Flip<dim>>, u8"Flip axis.")
    .add_property("axis_nr", &getFlipDirNr<Flip<dim>>, &setFlipDir<Flip<dim>>, u8"Number of the flip axis.")
    ;
}

DECLARE_GEOMETRY_ELEMENT_23D(Mirror, "Mirror",
    u8"Transfer that mirrors the geometry object along the specified  axis\n"
    u8"(",u8" version).\n\n"
    u8"Args:\n"
    u8"   axis (float or str): Flip axis number or name.\n"
    u8"   item (GeometryObject): Geometry object to flip.\n\n"
    u8"The effect of this transform is an original object with an added mirror\n"
    u8"reflection. The mirroring is done with respect to the axis, so the whole\n"
    u8"object should be bouned within one half-plane of its local coordinate\n"
    u8"system.\n\n"
    u8"The difference between the mirror and a flip is that the flip replaces the\n"
    u8"original object with its flipped version.\n\n"
    u8"See also:\n"
    u8"   :class:`plask.geometry.Flip`")
{
    GEOMETRY_ELEMENT_23D(Mirror, GeometryObjectTransform<dim>, py::no_init)
    .def("__init__", py::make_constructor(&Mirror_constructor1<Mirror<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .def("__init__", py::make_constructor(&Mirror_constructor2<Mirror<dim>>, py::default_call_policies(), (py::arg("axis"), py::arg("item")=shared_ptr<GeometryObjectD<dim>>())))
    .add_property("axis", &getFlipDir<Mirror<dim>>, &setFlipDir<Mirror<dim>>, u8"Mirror axis.")
    .add_property("axis_nr", &getFlipDirNr<Mirror<dim>>, &setFlipDir<Mirror<dim>>, u8"Number of the mirror axis.")
    ;
}

template <int dim>
shared_ptr<Clip<dim>> Clip_constructor1(shared_ptr<GeometryObjectD<dim>> object, const typename Primitive<dim>::Box& clip_box) {
    return plask::make_shared<Clip<dim>>(object, clip_box);
}

template <int dim> struct Clip_constructor2 {};
template <> struct Clip_constructor2<2> {
    static inline shared_ptr<Clip<2>> call(shared_ptr<GeometryObjectD<2>> object, const py::object& pleft, const py::object& pbottom, const py::object& pright, const py::object& ptop) {
            double left = pleft.is_none()? -INFINITY : py::extract<double>(pleft),
                   bottom = pbottom.is_none()? -INFINITY : py::extract<double>(pbottom),
                   right = pright.is_none()? INFINITY : py::extract<double>(pright),
                   top = ptop.is_none()? INFINITY : py::extract<double>(ptop);
            return plask::make_shared<Clip<2>>(object, Box2D(left, bottom, right, top));
    }
    const static py::detail::keywords<5> args;
};
const py::detail::keywords<5> Clip_constructor2<2>::args = (py::arg("item"),
                                                            py::arg("left")=py::object(), py::arg("bottom")=py::object(),
                                                            py::arg("right")=py::object(), py::arg("top")=py::object());

template <> struct Clip_constructor2<3> {
    static inline shared_ptr<Clip<3>> call(shared_ptr<GeometryObjectD<3>> object, const py::object& pback, const py::object& pleft, const py::object& pbottom, const py::object& pfront, const py::object& pright, const py::object& ptop) {
            double back = pback.is_none()? -INFINITY : py::extract<double>(pback),
                   left = pleft.is_none()? -INFINITY : py::extract<double>(pleft),
                   bottom = pbottom.is_none()? -INFINITY : py::extract<double>(pbottom),
                   front = pfront.is_none()? INFINITY : py::extract<double>(pfront),
                   right = pright.is_none()? INFINITY : py::extract<double>(pright),
                   top = ptop.is_none()? INFINITY : py::extract<double>(ptop);
            return plask::make_shared<Clip<3>>(object, Box3D(back, left, bottom, front, right, top));
    }
    const static py::detail::keywords<7> args;
};
const py::detail::keywords<7> Clip_constructor2<3>::args = (py::arg("item"),
                                                            py::arg("back")=py::object(), py::arg("left")=py::object(), py::arg("bottom")=py::object(),
                                                            py::arg("front")=py::object(), py::arg("right")=py::object(), py::arg("top")=py::object());


template <int dim> inline const char* ClipName();
template <> inline const char* ClipName<2>() { return "Clip2D"; }
template <> inline const char* ClipName<3>() { return "Clip3D"; }

template <int dim> inline const char* ClipDoc();
template <> inline const char* ClipDoc<2>() { return
    u8"Clip2D(item, box)\n"
    u8"Clip2D(item, left, bottom, right, top)\n\n"
    u8"Transform that clips the held geometry object to the specified clip-box\n"
    u8"(2D version).\n\n"
    u8"This transform is used to limit the size of any complicated geometry object to\n"
    u8"the specified rectangular box. This way, you can easily change e.g. a circle\n"
    u8"to a half- or quarter-circle. In order to use this transform, you must\n"
    u8"explicitly specify the coordinates of the clip-box in the local coordinates of\n"
    u8"the clipped object. However, the original object is never expanded, co you can\n"
    u8"freely make the box very large, or even infinite (which means no clipping at\n"
    u8"this side).\n\n"
    u8"Args:\n"
    u8"    item (GeometryObject2D): Object to clip.\n"
    u8"    left (float): Left side of the clipping box. *inf* by default.\n"
    u8"    bottom (float): Bottom side of the clipping box. *inf* by default.\n"
    u8"    right (float): Right side of the clipping box. *inf* by default.\n"
    u8"    top (float): Top side of the clipping box. *inf* by default.\n"
    u8"    box (Box2D): Clipping box.\n\n"
    u8"Example:\n"
    u8"    To make a half-circle with the flat bottom:\n\n"
    u8"    >>> circle = plask.geometry.Circle(2, 'GaAs')\n"
    u8"    >>> half_circle = plask.geometry.Clip2D(circle, bottom=0)\n"
    ;
}
template <> inline const char* ClipDoc<3>() { return
    u8"Clip3D(item, box)\n"
    u8"Clip3D(item, back, left, bottom, front, right, top)\n\n"
    u8"Transform that clips the held geometry object to the specified clip-box\n"
    u8"(3D version).\n\n"
    u8"This transform is used to limit the size of any complicated geometry object to\n"
    u8"the specified rectangular box. This way, you can easily change e.g. a cylinder\n"
    u8"to a half- or quarter-cilinder. In order to use this transform, you must\n"
    u8"explicitly specify the coordinates of the clip-box in the local coordinates of\n"
    u8"the clipped object. However, the original object is never expanded, co you can\n"
    u8"freely make the box very large, or even infinite (which means no clipping at\n"
    u8"this side).\n\n"
    u8"Args:\n"
    u8"    item (GeometryObject3D): Object to clip.\n"
    u8"    back (float): Back side of the clipping box. *inf* by default.\n"
    u8"    left (float): Left side of the clipping box. *inf* by default.\n"
    u8"    bottom (float): Bottom side of the clipping box. *inf* by default.\n"
    u8"    front (float): Front side of the clipping box. *inf* by default.\n"
    u8"    right (float): Right side of the clipping box. *inf* by default.\n"
    u8"    top (float): Top side of the clipping box. *inf* by default.\n"
    u8"    box (Box3D): Clipping box.\n\n"
    u8"Example:\n"
    u8"    To make a half-cylinder with the flat front side:\n\n"
    u8"    >>> cylinder = plask.geometry.Cylinder(2, 1, 'GaAs')\n"
    u8"    >>> half_cylinder = plask.geometry.Clip3D(cylinder, front=0)\n"
    ;
}

template <int dim>
inline static void init_Clip()
{
    py::class_<Clip<dim>, shared_ptr<Clip<dim>>, py::bases<GeometryObjectTransform<dim>>, boost::noncopyable>(
    ClipName<dim>(), ClipDoc<dim>(), py::no_init)
        .def("__init__", py::make_constructor(&Clip_constructor1<dim>, py::default_call_policies(), (py::arg("item"), py::arg("box"))))
        .def("__init__", py::make_constructor(&Clip_constructor2<dim>::call, py::default_call_policies(), Clip_constructor2<dim>::args))
        .def_readwrite("clipbox", &Clip<dim>::clipBox, "Clipping box.")
        ;
}

template <int dim>
shared_ptr<Intersection<dim>> Intersection_constructor(shared_ptr<GeometryObjectD<dim>> object, shared_ptr<GeometryObjectD<dim>> shape) {
    return plask::make_shared<Intersection<dim>>(object, shape);
}

template <int dim> inline const char* IntersectionName();
template <> inline const char* IntersectionName<2>() { return "Intersection2D"; }
template <> inline const char* IntersectionName<3>() { return "Intersection3D"; }

template <int dim> inline const char* IntersectionDoc();
template <> inline const char* IntersectionDoc<2>() { return
    u8"Intersection2D(item, shape)\n\n"
    u8"Transform that clips the held geometry object to the specified envelope\n"
    u8"(2D version).\n\n"
    u8"This transform is a more advanced version of :class:`~plask.geometry.Clip2D`.\n"
    u8"Instead of a simple box, you can specify any geometry object as a clipping\n"
    u8"envelope. The material of the evelope is ignored.\n\n"
    u8"Args:\n"
    u8"    item (GeometryObject2D): Object to clip.\n"
    u8"    shape (GeometryObject2D): Object to serve as a clipping envelope.\n\n"
//TODO
//    "Example:\n"
//    "    To make a half-circle with the flat bottom:\n\n"
//    "    >>> circle = plask.geometry.Circle(2, 'GaAs')\n"
//    "    >>> half_circle = plask.geometry.Intersection2D(circle, bottom=0)\n"
    ;
}
template <> inline const char* IntersectionDoc<3>() { return
    u8"Intersection3D(item, shape)\n\n"
    u8"Transform that clips the held geometry object to the specified envelope\n"
    u8"(3D version).\n\n"
    u8"This transform is a more advanced version of :class:`~plask.geometry.Clip3D`.\n"
    u8"Instead of a simple box, you can specify any geometry object as a clipping\n"
    u8"envelope. The material of the evelope is ignored.\n\n"
    u8"Args:\n"
    u8"    item (GeometryObject3D): Object to clip.\n"
    u8"    shape (GeometryObject3D): Object to serve as a clipping envelope.\n\n"
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
                      u8"Clipping envelope.\n\n"
                      u8"This is a geometry object that serves as a clipping envelope. The main item\n"
                      u8"of this transform is clipped to the shape of the envelope.")
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
