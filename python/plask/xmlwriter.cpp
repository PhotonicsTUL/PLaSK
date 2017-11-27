#include "python_globals.h"
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>

#include <plask/config.h>
#include <plask/geometry/space.h>
#include <plask/mesh/mesh.h>
#include <plask/utils/string.h>

#include "python_util/raw_constructor.h"

namespace plask { namespace python {

struct XplWriter
{
    py::object geometry, mesh, names;

  protected:

    struct PythonWriteXMLCallback: public GeometryObject::WriteXMLCallback {
        XplWriter* writer;
        PythonWriteXMLCallback(XplWriter* writer): writer(writer) {}
        std::string getName(const GeometryObject& object, AxisNames& axesNames) const override {
            py::stl_input_iterator<std::string> end;
            for (auto name = py::stl_input_iterator<std::string>(writer->geometry); name != end; ++name)
                if ((const Geometry*)py::extract<Geometry*>(writer->geometry[*name]) == &object) return *name;
            for (auto name = py::stl_input_iterator<std::string>(writer->names); name != end; ++name)
                if ((const GeometryObject*)py::extract<GeometryObject*>(writer->names[*name]) == &object) return *name;
            return "";
        }
    };


    struct PythonOutput: public XMLWriter::Output {
        py::object pyfile;
        PythonOutput(const py::object& pyfile): pyfile(pyfile) {}
        void write(const char* buffer, std::size_t n) override {
            /*char tmp[n+1];
            tmp[n] = 0;
            std::copy_n(buffer, n, tmp);
            pyfile.attr("write")((const char*)tmp);*/
            pyfile.attr("write")(std::string(buffer, n));
        }
    };

    void saveGeometry(XMLWriter& writer) {
        auto tag = writer.addTag("geometry");
        PythonWriteXMLCallback callback(this);
        py::stl_input_iterator<std::string> begin(geometry), end;
        for (auto key = begin; key != end; ++key) {
            shared_ptr<Geometry> val = py::extract<shared_ptr<Geometry>>(geometry[*key]);
            val->writeXML(tag, callback, val->axisNames);
        }
    }

    void saveMesh(XMLWriter& writer) {
        auto tag = writer.addTag("grids");
        py::stl_input_iterator<std::string> begin(mesh), end;
        for (auto key = begin; key != end; ++key) {
            shared_ptr<Mesh> val = py::extract<shared_ptr<Mesh>>(mesh[*key]);
            val->writeXML(tag.addTag("mesh").attr("name", *key));
        }
    }

  public:

    XplWriter(const py::object& geo, const py::object& msh, const py::object& nams):
        geometry(geo), mesh(msh), names(nams)
    {
        if (geometry == py::object()) geometry = py::dict();
        if (mesh == py::object()) mesh = py::dict();
        if (names == py::object()) names = py::dict();
    }

    std::string __str__() {
        std::stringstream ss;
        {
            XMLWriter writer(ss);
            auto plask = writer.addTag("plask");
            saveGeometry(writer);
            saveMesh(writer);
        }
        return ss.str();
    }

    void saveto(const py::object& target) {
        py::extract<std::string> fname(target);
        if (fname.check()) {
            XMLWriter writer(fname);
            writer.writeHeader();
            auto plask = writer.addTag("plask");
            saveGeometry(writer);
            saveMesh(writer);
        } else {
            XMLWriter writer(new PythonOutput(target));
            writer.writeHeader();
            auto plask = writer.addTag("plask");
            saveGeometry(writer);
            saveMesh(writer);
        }
    }
};

XplWriter* XmlWriter(const py::object& geo, const py::object& msh, const py::object& nams) {
    writelog(LOG_WARNING, "'XmlWriter' class has been renamed to 'XplWriter'. Please update your code!");
    return new XplWriter(geo, msh, nams);
}

void register_xml_writer()
{
    py::class_<XplWriter>("XplWriter",
                          u8"XPL writer that can save existing geometries and meshes to the XPL.\n\n"
                          u8"Objects of this class contain three dictionaries:\n"
                          u8":attr:`~plask.XplWriter.geometry` and :attr:`~plask.XplWriter.mesh`\n"
                          u8"that should contain the geometries or meshes, which should be saved and\n"
                          u8":attr:`~plask.XplWriter.names` with other geometry objects that should be\n"
                          u8"explicitly named in the resulting XPL. All these dictionaries must have strings\n"
                          u8"as their keys and corresponding objects as values.\n\n"
                          u8"Args:\n"
                          u8"    geo (dict): Dictionary with geometries that should be saved to the file.\n"
                          u8"    mesh (dict): Dictionary with meshes that should be saved to the file.\n"
                          u8"    names (dict): Dictionary with names of the geometry objects that should be\n"
                          u8"                  explicitly named in the file.\n\n"
                          u8"The final XPL can be simply retrieved as a string (``str(writer)``) or saved to\n"
                          u8"a file with the :meth:`~plask.XplWriter.saveto` method.\n\n"
                          u8"Example:\n"
                          u8"    Create an XML file with a simple geometry:\n\n"
                          u8"    >>> rect = plask.geometry.Rectangle(2, 1, 'GaAs')\n"
                          u8"    >>> geo = plask.geometry.Cartesian2D(rect)\n"
                          u8"    >>> xml = plask.XplWriter({'geo': geo}, {}, {'rect': rect})\n"
                          u8"    >>> print(xml)\n"
                          u8"    <plask>\n"
                          u8"      <geometry>\n"
                          u8"        <cartesian2d name=\"geo\" axes=\"zxy\">\n"
                          u8"          <extrusion length=\"inf\">\n"
                          u8"            <block2d name=\"rect\" material=\"GaAs\" dx=\"2\" dy=\"1\"/>\n"
                          u8"          </extrusion>\n"
                          u8"        </cartesian2d>\n"
                          u8"      </geometry>\n"
                          u8"      <grids/>\n"
                          u8"    </plask>\n",
        py::init<py::object,py::object,py::object>((py::arg("geo")=py::object(), py::arg("msh")=py::object(), py::arg("names")=py::object())))
        .def("__str__", &XplWriter::__str__)
        .def("saveto", &XplWriter::saveto, py::arg("target"),
            u8"Save the resulting XPL to the file.\n\n"
            u8"Args:\n"
            u8"    target (string or file): A file name or an open file object to save to.\n")
        .def_readwrite("geometry", &XplWriter::geometry, "Dictionary with geometries that should be saved to the file.")
        .def_readwrite("mesh", &XplWriter::mesh, u8"Dictionary with meshes that should be saved to the file.")
        .def_readwrite("names", &XplWriter::names, u8"Dictionary with names of the geometry objects that should be explicitly named\n"
                                                   u8"in the file.")
    ;

    py::def("XmlWriter", &XmlWriter, py::return_value_policy<py::manage_new_object>(),
            (py::arg("geo")=py::object(), py::arg("msh")=py::object(), py::arg("names")=py::object()));
}

}} // namespace plask::python
