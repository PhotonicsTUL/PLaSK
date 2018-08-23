#include "solver3d.h"
#include "expansion3d.h"
#include "../diagonalizer.h"

namespace plask { namespace optical { namespace slab {

FourierSolver3D::FourierSolver3D(const std::string& name): SlabSolver<SolverOver<Geometry3D>>(name),
    size_long(12), size_tran(12),
    klong(0.), ktran(0.),
    symmetry_long(Expansion::E_UNSPECIFIED), symmetry_tran(Expansion::E_UNSPECIFIED),
    dct(2),
    expansion(this),
    refine_long(16), refine_tran(16),
    oversampling_long(1.), oversampling_tran(1.)
{
    pml_tran.factor = {1., -2.};
    pml_long.factor = {1., -2.};
    smooth = 0.00025;
}

static inline void updatePML(PML& pml, XMLReader& reader) {
    pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
    pml.size = reader.getAttribute<double>("size", pml.size);
    pml.dist = reader.getAttribute<double>("dist", pml.dist);
    pml.order = reader.getAttribute<double>("shape", pml.order);
}

template <typename T>
static inline void readComaAttr(XMLReader& reader, const std::string& attr, T& long_field, T& tran_field, bool nocommon=false) {
    if (reader.hasAttribute(attr)) {
        std::string value = reader.requireAttribute<std::string>(attr);
        if (value.find(',') == std::string::npos) {
            if (nocommon) throw XMLBadAttrException(reader, attr, value);
            long_field = tran_field = boost::lexical_cast<T>(value);
        } else {
            auto values = splitString2(value, ',');
            long_field = boost::lexical_cast<T>(values.first);
            tran_field = boost::lexical_cast<T>(values.second);
        }
        if (reader.hasAttribute(attr+"-long")) throw XMLConflictingAttributesException(reader, attr, attr+"-long");
        if (reader.hasAttribute(attr+"-tran")) throw XMLConflictingAttributesException(reader, attr, attr+"-tran");
    } else {
        long_field = reader.getAttribute<T>(attr+"-long", long_field);
        tran_field = reader.getAttribute<T>(attr+"-tran", tran_field);
    }
}

static inline Expansion::Component readSymmetry(const FourierSolver3D* solver, const XMLReader& reader, const std::string& repr) {
    AxisNames* axes = nullptr;
    if (solver->getGeometry()) axes = &solver->getGeometry()->axisNames;
    if (repr == "none" || repr == "NONE" || repr == "None")
        return Expansion::E_UNSPECIFIED;
    else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
        return Expansion::E_TRAN;
    else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
        return Expansion::E_LONG;
    else
        throw XMLBadAttrException(reader, "symmetry", repr, "symmetric field component name (maybe you need to specify the geometry first)");
}

void FourierSolver3D::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            readComaAttr(reader, "size", size_long, size_tran);
            readComaAttr(reader, "refine", refine_long, refine_tran);
            readComaAttr(reader, "oversampling", oversampling_long, oversampling_tran);
            smooth = reader.getAttribute<double>("smooth", smooth);
            int dc = reader.getAttribute<int>("dct", dct);
            if (dc != 1 && dc != 2)
                throw XMLBadAttrException(reader, "dct", boost::lexical_cast<std::string>(dc), "\"1\" or \"2\"");
            dct = dc;
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
            lam0 = reader.getAttribute<double>("lam0", NAN);
            always_recompute_gain = reader.getAttribute<bool>("update-gain", always_recompute_gain);
            max_temp_diff = reader.getAttribute<double>("temp-diff", max_temp_diff);
            temp_dist = reader.getAttribute<double>("temp-dist", temp_dist);
            temp_layer = reader.getAttribute<double>("temp-layer", temp_layer);
            reader.requireTagEnd();
        } else if (param == "interface") {
            if (reader.hasAttribute("index")) {
                throw XMLException(reader, "Setting interface by layer index is not supported anymore (set it by object or position)");
            } else if (reader.hasAttribute("position")) {
                if (reader.hasAttribute("object")) throw XMLConflictingAttributesException(reader, "index", "object");
                if (reader.hasAttribute("path")) throw XMLConflictingAttributesException(reader, "index", "path");
                setInterfaceAt(reader.requireAttribute<double>("position"));
            } else if (reader.hasAttribute("object")) {
                auto object = manager.requireGeometryObject<GeometryObject>(reader.requireAttribute("object"));
                PathHints path; if (auto pathattr = reader.getAttribute("path")) path = manager.requirePathHints(*pathattr);
                setInterfaceOn(object, path);
            } else if (reader.hasAttribute("path")) {
                throw XMLUnexpectedAttrException(reader, "path");
            }
            reader.requireTagEnd();
        } else if (param == "vpml") {
            vpml.factor = reader.getAttribute<dcomplex>("factor", vpml.factor);
            vpml.size = reader.getAttribute<double>("size", vpml.size);
            vpml.dist = reader.getAttribute<double>("dist", vpml.dist);
            if (reader.hasAttribute("order")) { //TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <vpml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
                vpml.order = reader.requireAttribute<double>("order");
            }
            vpml.order = reader.getAttribute<double>("shape", vpml.order);
            reader.requireTagEnd();
        } else if (param == "transfer") {
            transfer_method = reader.enumAttribute<Transfer::Method>("method")
                .value("auto", Transfer::METHOD_AUTO)
                .value("reflection", Transfer::METHOD_REFLECTION)
                .value("admittance", Transfer::METHOD_ADMITTANCE)
                .get(transfer_method);
            reader.requireTagEnd();
        } else if (param == "pmls") {
            updatePML(pml_long, reader);
            updatePML(pml_tran, reader);
            while (reader.requireTagOrEnd()) {
                std::string node = reader.getNodeName();
                if (node == "long") {
                    updatePML(pml_long, reader);
                    reader.requireTagEnd();
                } else if (node == "tran") {
                    updatePML(pml_tran, reader);
                    reader.requireTagEnd();
                } else throw XMLUnexpectedElementException(reader, "<tran>, <long>, or </pmls>", node);
            }
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                                .value("undefined", EMISSION_UNSPECIFIED)
                                .value("top", EMISSION_TOP)
                                .value("bottom", EMISSION_BOTTOM)
                       .get(emission);
            k0 = 2e3*PI / reader.getAttribute<dcomplex>("wavelength", 2e3*PI / k0);
            ktran = reader.getAttribute<dcomplex>("k-tran", ktran);
            klong = reader.getAttribute<dcomplex>("k-long", klong);
            std::string sym_tran, sym_long;
            readComaAttr(reader, "symmetry", sym_long, sym_tran, true);
            if (sym_long != "") setSymmetryLong(readSymmetry(this, reader, sym_long));
            if (sym_tran != "") setSymmetryTran(readSymmetry(this, reader, sym_tran));
            reader.requireTagEnd();
        } else if (param == "root") {
            readRootDiggerConfig(reader);
        } else
            parseStandardConfiguration(reader, manager);
    }
}


void FourierSolver3D::onInitialize()
{
    this->setupLayers();
    if (this->interface == -1)
        Solver::writelog(LOG_DETAIL, "Initializing Fourier3D solver ({0} layers in the stack)",
                                     this->stack.size());
    else
        Solver::writelog(LOG_DETAIL, "Initializing Fourier3D solver ({0} layers in the stack, interface after {1} layer{2})",
                                     this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    setExpansionDefaults();
    expansion.init();
    this->recompute_integrals = true;
}


void FourierSolver3D::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t FourierSolver3D::findMode(FourierSolver3D::What what, dcomplex start)
{
    expansion.setSymmetryLong(symmetry_long);
    expansion.setSymmetryTran(symmetry_tran);
    expansion.setLam0(this->lam0);
    Solver::initCalculation();
    ensureInterface();
    if (!transfer) initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case FourierSolver3D::WHAT_WAVELENGTH:
            expansion.setKlong(klong);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) { expansion.setK0(2e3*PI/x); return transfer->determinant(); }, "lam");
            break;
        case FourierSolver3D::WHAT_K0:
            expansion.setKlong(klong);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) { expansion.setK0(x); return transfer->determinant(); }, "k0");
            break;
        case FourierSolver3D::WHAT_KLONG:
            expansion.setK0(this->k0);
            expansion.setKtran(ktran);
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            root = getRootDigger([this](const dcomplex& x) { expansion.klong = x; return transfer->determinant(); }, "klong");
            break;
        case FourierSolver3D::WHAT_KTRAN:
            expansion.setK0(this->k0);
            expansion.setKlong(klong);
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            root = getRootDigger([this](const dcomplex& x) { expansion.ktran = x; return transfer->determinant(); }, "ktran");
            break;
    }
    root->find(start);
    return insertMode();
}




LazyData<Vec<3,dcomplex>> FourierSolver3D::getE(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightE", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldE(modes[num].power, dst_mesh, method);
}


LazyData<Vec<3,dcomplex>> FourierSolver3D::getH(size_t num, shared_ptr<const MeshD<3>> dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightH", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldH(modes[num].power, dst_mesh, method);
}


LazyData<double> FourierSolver3D::getMagnitude(size_t num, shared_ptr<const MeshD<3> > dst_mesh, InterpolationMethod method)
{
    if (num >= modes.size()) throw BadInput(this->getId()+".outLightMagnitude", "Mode {0} has not been computed", num);
    assert(transfer);
    applyMode(modes[num]);
    return transfer->getFieldMagnitude(modes[num].power, dst_mesh, method);
}


cvector FourierSolver3D::incidentVector(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam) {
    bool changed = Solver::initCalculation() || setExpansionDefaults(isnan(lam));
    if (!isnan(lam)) {
        dcomplex k0 = 2e3*M_PI / lam;
        if (!is_zero(k0 - expansion.getK0())) {
            expansion.setK0(k0);
            changed = true;
        }
    }
    size_t layer = stack[(side == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1];
    if (!transfer) {
        initTransfer(expansion, true);
        changed = true;
    }
    if (changed) {
        transfer->initDiagonalization();
        transfer->diagonalizer->diagonalizeLayer(layer);
    } else if (!transfer->diagonalizer->isDiagonalized(layer))
        transfer->diagonalizer->diagonalizeLayer(layer);

    if (polarization == ExpansionPW3D::E_UNSPECIFIED)
        throw BadInput(getId(), "Unspecified incident polarization for reflectivity computation");
    if (expansion.symmetry_long == Expansion::Component(3-polarization))
        throw BadInput(getId(), "Current longitudinal symmetry is inconsistent with the specified incident polarization");
    if (expansion.symmetry_tran == Expansion::Component(3-polarization))
        throw BadInput(getId(), "Current transverse symmetry is inconsistent with the specified incident polarization");
    size_t idx = (polarization == ExpansionPW3D::E_LONG)? expansion.iEx(0,0) : expansion.iEy(0,0);
    cvector incident(expansion.matrixSize(), 0.);
    incident[idx] = 1.;
    return transfer->diagonalizer->invTE(layer) * incident;
}

}}} // namespace
