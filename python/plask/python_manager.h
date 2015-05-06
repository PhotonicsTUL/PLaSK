#ifndef PLASK__PYTHON_MANAGER_H
#define PLASK__PYTHON_MANAGER_H

#include <plask/manager.h>

namespace plask { namespace  python {


struct PLASK_PYTHON_API PythonManager: public Manager {

//     /// List of constant profiles
//     py::dict profiles;

    /// Locals read from &lt;defines&gt; section and supplied by user
    py::dict locals;

    MaterialsDB* materialsDB;

    PythonManager(MaterialsDB* db=nullptr, bool draft=false): materialsDB(db? db : &MaterialsDB::getDefault()) {
        this->draft = draft;
    }

    virtual ~PythonManager() {}

    shared_ptr<Solver> loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name) override;

    void loadDefines(XMLReader& reader) override;

    void loadConnects(XMLReader& reader) override;

    void loadMaterial(XMLReader& reader, MaterialsDB& materialsDB) override;

    void loadMaterialModule(XMLReader& reader, MaterialsDB& materialsDB);

    void loadMaterials(XMLReader& reader, MaterialsDB& materialsDB) override;

    static void export_dict(py::object self, py::dict dict);

    void loadScript(XMLReader& reader) override;

  private:
    void removeSpaces(unsigned xmlline);
};

}} // namespace plask::python

#endif // PLASK__PYTHON_MANAGER_H

