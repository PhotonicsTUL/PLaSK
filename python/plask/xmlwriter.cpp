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
                          "XPL writer that can save existing geometries and meshes to the XPL.\n\n"
                          "Objects of this class contain three dictionaries:\n"
                          ":attr:`~plask.XplWriter.geometry` and :attr:`~plask.XplWriter.mesh`\n"
                          "that should contain the geometries or meshes, which should be saved and\n"
                          ":attr:`~plask.XplWriter.names` with other geometry objects that should be\n"
                          "explicitly named in the resulting XPL. All these dictionaries must have strings\n"
                          "as their keys and corresponding objects as values.\n\n"
                          "Args:\n"
                          "    geo (dict): Dictionary with geometries that should be saved to the file.\n"
                          "    mesh (dict): Dictionary with meshes that should be saved to the file.\n"
                          "    names (dict): Dictionary with names of the geometry objects that should be\n"
                          "                  explicitly named in the file.\n\n"
                          "The final XPL can be simply retrieved as a string (``str(writer)``) or saved to\n"
                          "a file with the :meth:`~plask.XplWriter.saveto` method.\n\n"
                          "Example:\n"
                          "    Create an XML file with a simple geometry:\n\n"
                          "    >>> rect = plask.geometry.Rectangle(2, 1, 'GaAs')\n"
                          "    >>> geo = plask.geometry.Cartesian2D(rect)\n"
                          "    >>> xml = plask.XplWriter({'geo': geo}, {}, {'rect': rect})\n"
                          "    >>> print(xml)\n"
                          "    <plask>\n"
                          "      <geometry>\n"
                          "        <cartesian2d name=\"geo\" axes=\"zxy\">\n"
                          "          <extrusion length=\"inf\">\n"
                          "            <block2d name=\"rect\" material=\"GaAs\" dx=\"2\" dy=\"1\"/>\n"
                          "          </extrusion>\n"
                          "        </cartesian2d>\n"
                          "      </geometry>\n"
                          "      <grids/>\n"
                          "    </plask>\n",
        py::init<py::object,py::object,py::object>((py::arg("geo")=py::object(), py::arg("msh")=py::object(), py::arg("names")=py::object())))
        .def("__str__", &XplWriter::__str__)
        .def("saveto", &XplWriter::saveto, py::arg("target"),
            "Save the resulting XPL to the file.\n\n"
            "Args:\n"
            "    target (string or file): A file name or an open file object to save to.\n")
        .def_readwrite("geometry", &XplWriter::geometry, "Dictionary with geometries that should be saved to the file.")
        .def_readwrite("mesh", &XplWriter::mesh, "Dictionary with meshes that should be saved to the file.")
        .def_readwrite("names", &XplWriter::names, "Dictionary with names of the geometry objects that should be explicitly named\n"
                                                   "in the file.")
    ;

    py::def("XmlWriter", &XmlWriter, py::return_value_policy<py::manage_new_object>(),
            (py::arg("geo")=py::object(), py::arg("msh")=py::object(), py::arg("names")=py::object()));
}

}} // namespace plask::python
