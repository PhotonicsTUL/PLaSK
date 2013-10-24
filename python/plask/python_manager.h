#ifndef PLASK__PYTHON_MANAGER_H
#define PLASK__PYTHON_MANAGER_H

#include <plask/manager.h>

namespace plask { namespace  python {


struct PythonManager: public Manager {

    /// List of constant profiles
    py::dict profiles;

    /// Locals read from &lt;defines&gt; section and supplied by user
    py::dict locals;

    MaterialsDB* materialsDB;

    PythonManager(MaterialsDB* db=nullptr) : materialsDB(db? db : &MaterialsDB::getDefault()) {}

    virtual ~PythonManager() {}

    virtual shared_ptr<Solver> loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name) override;

    virtual void loadDefines(XMLReader& reader) override;

    virtual void loadConnects(XMLReader& reader) override;

    virtual void loadMaterials(XMLReader& reader, shared_ptr<const MaterialsSource> materialsSource) override;

    static void export_dict(py::object self, py::dict dict);

    virtual void loadScript(XMLReader& reader) override;

    // static std::string removeSpaces(const std::string& source);
};

}} // namespace plask::python

#endif // PLASK__PYTHON_MANAGER_H

