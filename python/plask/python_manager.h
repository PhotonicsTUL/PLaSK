#ifndef PLASK__PYTHON_MANAGER_H
#define PLASK__PYTHON_MANAGER_H

#include <plask/manager.h>

namespace plask { namespace  python {


struct PythonManager: public Manager {

    MaterialsDB* materialsDB;

    PythonManager(MaterialsDB* db=nullptr) : materialsDB(db? db : &MaterialsDB::getDefault()) {}

    void read(py::object src);

    virtual shared_ptr<Solver> loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name);

    virtual void loadRelations(XMLReader& reader);

    virtual void loadMaterials(XMLReader& reader, MaterialsDB& materialsDB = MaterialsDB::getDefault());

    static void export_dict(py::object self, py::dict dict);

    // static std::string removeSpaces(const std::string& source);
};

}} // namespace plask::python

#endif // PLASK__PYTHON_MANAGER_H

