#include "fdtd.hpp"
namespace plask { namespace solvers { namespace optical_fdtd {

FDTDSolver::FDTDSolver(const std::string& name)
    : plask::SolverOver<plask::Geometry2DCartesian>(name),
      baseTimeUnit(1 / (plask::phys::c * 1e-9)),
      fieldMesh(plask::make_shared<RectangularMesh2D>()),
      outLightH(this, &FDTDSolver::getLightH),
      outLightE(this, &FDTDSolver::getLightE) {}

std::string FDTDSolver::getClassName() const { return "optical.FDTD2D"; }

void FDTDSolver::compute(double time) {
    initCalculation();
    long long int n = timeToSteps(time);  // Compute the neccesary timesteps to achieve the given time
    writelog(LOG_INFO, "The simulation will take " + std::to_string(n) + " steps");
    for (size_t i = 0; i < n; i++) step();
    elapsedTime += (dt * baseTimeUnit * n);
}

void FDTDSolver::step() {
    initCalculation();
    compFields->step();
}

void FDTDSolver::getField(const std::vector<std::string>& fields) {
    storedFieldComponents = fields;
    compFields->loop_in_chunks(&solver_chunkloop, this, compFields->total_volume());
}

void FDTDSolver::chunkloop(meep::fields_chunk* fc,
                           int ichunk,
                           meep::component cgrid,
                           meep::ivec is,
                           meep::ivec ie,
                           meep::vec s0,
                           meep::vec s1,
                           meep::vec e0,
                           meep::vec e1,
                           double dV0,
                           double dV1,
                           meep::ivec shift,
                           std::complex<double> shift_phase,
                           const meep::symmetry& S,
                           int sn,
                           void* chunkloop_data) {
    meep::vec rshift(shift * (0.5 * fc->gv.inva));
    std::vector<std::string> fields = ((FDTDSolver*)chunkloop_data)->storedFieldComponents;

    auto dx = 1, dy = 1;
    if (((FDTDSolver*)chunkloop_data)->compBounds.little_corner().x() < 0) {
        dx = ((FDTDSolver*)chunkloop_data)->compBounds.little_corner().x() + 1;
    }
    if (((FDTDSolver*)chunkloop_data)->compBounds.little_corner().y() < 0) {
        dy = ((FDTDSolver*)chunkloop_data)->compBounds.little_corner().y() + 1;
    }

    LOOP_OVER_IVECS(fc->gv, is, ie, idx) {
        using namespace meep;
        IVEC_LOOP_ILOC(fc->gv, iparent);
        // IVEC_LOOP_LOC macro copied because of the plask and meep vec conflict in the next line, so it had to be specified
        // explicitly
        meep::vec rparent((fc->gv).dim);
        rparent.set_direction(direction(loop_d1), (0.5 * loop_is1 + loop_i1) * (fc->gv).inva);
        rparent.set_direction(direction(loop_d2), (0.5 * loop_is2 + loop_i2) * (fc->gv).inva);
        rparent.set_direction(direction(loop_d3), (0.5 * loop_is3 + loop_i3) * (fc->gv).inva);
        ivec ichild = S.transform(iparent, sn) + shift;
        meep::vec rchild = S.transform(rparent, sn) + rshift;

        plask::Vec<3, dcomplex> tmp_vec{fc->get_field(ComponentMap.at(fields[0]), ichild),
                                        fc->get_field(ComponentMap.at(fields[1]), ichild),
                                        fc->get_field(ComponentMap.at(fields[2]), ichild)};
        auto x = (ichild.x() - dx) / 2;
        auto y = (ichild.y() - dy) / 2;
        auto index = ((FDTDSolver*)chunkloop_data)->fieldMesh->index(x, y);
        ((FDTDSolver*)chunkloop_data)->fieldValues[index] = tmp_vec;
    }
};

void FDTDSolver::solver_chunkloop(meep::fields_chunk* fc,
                                  int ichunk,
                                  meep::component cgrid,
                                  meep::ivec is,
                                  meep::ivec ie,
                                  meep::vec s0,
                                  meep::vec s1,
                                  meep::vec e0,
                                  meep::vec e1,
                                  double dV0,
                                  double dV1,
                                  meep::ivec shift,
                                  std::complex<double> shift_phase,
                                  const meep::symmetry& S,
                                  int sn,
                                  void* chunkloop_data) {
    FDTDSolver* self = (FDTDSolver*)chunkloop_data;
    self->chunkloop(fc, ichunk, cgrid, is, ie, s0, s1, e0, e1, dV0, dV1, shift, shift_phase, S, sn, self);
}

void FDTDSolver::outputHDF5(std::string type) {
    initCalculation();
    compFields->output_hdf5(ComponentMap.at(type), compBounds.surroundings());
}

void FDTDSolver::loadConfiguration(XMLReader& reader, Manager& manager) {
    double ax = 0, ay = 0, bx = 0, by = 0, lam = 0, start_time = 0, slowness = 0, amplitude = 0, width = 0;
    std::string component, end_time;

    while (reader.requireTagOrEnd()) {
        std::string curr_node = reader.getNodeName();

        if (curr_node == "comp_space") {
            resolution = *reader.getAttribute<int>("resolution");
            layerPML = *reader.getAttribute<double>("pml");
            shiftPML = *reader.getAttribute<double>("pml_shift");
            reader.requireTagEnd();
        } else if (curr_node == "simulation_options") {
            courantFactor = *reader.getAttribute<double>("courant_factor");
            wavelength = *reader.getAttribute<double>("wavelength");
            temperature = *reader.getAttribute<double>("temperature");
            reader.requireTagEnd();
        } else if (curr_node == "sources") {
            while (reader.requireTagOrEnd()) {
                curr_node = reader.getNodeName();

                if (curr_node == "point") {
                    ax = *reader.getAttribute<double>("x");
                    ay = *reader.getAttribute<double>("y");
                    while (reader.requireTagOrEnd()) {
                        curr_node = reader.getNodeName();

                        if (curr_node == "continuous") {
                            lam = *reader.getAttribute<double>("wavelength");
                            start_time = *reader.getAttribute<double>("start-time");
                            end_time = *reader.getAttribute<std::string>("end-time");
                            component = *reader.getAttribute<std::string>("component");
                            slowness = *reader.getAttribute<double>("slowness");
                            amplitude = *reader.getAttribute<double>("amplitude");
                            sourceContainer.push_back(SourceMEEP("point_cont", ax, ay, bx, by, lam, start_time, end_time, amplitude,
                                                                 width, slowness, component));
                            reader.requireTagEnd();
                        } else if (curr_node == "gauss") {
                            lam = *reader.getAttribute<double>("wavelength");
                            start_time = *reader.getAttribute<double>("start-time");
                            end_time = *reader.getAttribute<std::string>("end-time");
                            width = *reader.getAttribute<double>("width");
                            component = *reader.getAttribute<std::string>("component");
                            amplitude = *reader.getAttribute<double>("amplitude");
                            sourceContainer.push_back(SourceMEEP("point_gauss", ax, ay, bx, by, lam, start_time, end_time,
                                                                 amplitude, width, slowness, component));
                            reader.requireTagEnd();
                        }
                    }
                } else if (curr_node == "volume") {
                    ax = *reader.getAttribute<double>("ax");
                    ay = *reader.getAttribute<double>("ay");
                    bx = *reader.getAttribute<double>("bx");
                    by = *reader.getAttribute<double>("by");
                    while (reader.requireTagOrEnd()) {
                        curr_node = reader.getNodeName();

                        if (curr_node == "continuous") {
                            lam = *reader.getAttribute<double>("wavelength");
                            start_time = *reader.getAttribute<double>("start-time");
                            end_time = *reader.getAttribute<std::string>("end-time");
                            component = *reader.getAttribute<std::string>("component");
                            slowness = *reader.getAttribute<double>("slowness");
                            amplitude = *reader.getAttribute<double>("amplitude");
                            sourceContainer.push_back(SourceMEEP("vol_cont", ax, ay, bx, by, lam, start_time, end_time, amplitude,
                                                                 width, slowness, component));
                            reader.requireTagEnd();
                        } else if (curr_node == "gauss") {
                            lam = *reader.getAttribute<double>("wavelength");
                            start_time = *reader.getAttribute<double>("start-time");
                            end_time = *reader.getAttribute<std::string>("end-time");
                            width = *reader.getAttribute<double>("width");
                            component = *reader.getAttribute<std::string>("component");
                            amplitude = *reader.getAttribute<double>("amplitude");
                            sourceContainer.push_back(SourceMEEP("vol_gauss", ax, ay, bx, by, lam, start_time, end_time, amplitude,
                                                                 width, slowness, component));
                            reader.requireTagEnd();
                        }
                    }
                }
            }
        } else
            parseStandardConfiguration(reader, manager, "<geometry>");
    }
}

double FDTDSolver::eps(const meep::vec& r) {
    plask::Vec<2, double> geomVector{r.x(), r.y()};
    double nr = geometry->getMaterial(geomVector)->nr(wavelength, temperature);
    return nr * nr;
}

double FDTDSolver::conductivity(const meep::vec& r) {
    plask::Vec<2, double> geomVector{r.x(), r.y()};
    plask::dcomplex complex_index = geometry->getMaterial(geomVector)->Nr(wavelength, temperature);
    return (2*plask::PI)*(1/(wavelength*1e-3))*(-complex_index.imag()/complex_index.real()); 
    //Tensor2<double> sigma_tensor = geometry->getMaterial(geomVector)->cond(temperature);
    //double meep_conductivity = (1e-6/plask::phys::c)*sigma_tensor.tran()*(1/(plask::phys::epsilon0*eps(r)));
    //return meep_conductivity;
}

/*
DFT Fluxes
*/

FluxDFT FDTDSolver::addFluxDFT(plask::Vec<2, double> p1, plask::Vec<2, double> p2, double wl_min, double wl_max, int Nlengths) {
    meep::volume flux_area(meep::vec(p1[0], p1[1]), meep::vec(p2[0], p2[1]));
    double freq_min = 1 / (wl_max * 1e-3), freq_max = 1 / (wl_min * 1e-3);
    shared_ptr<meep::dft_flux> meep_dft_flux = make_shared<meep::dft_flux>(
        compFields->add_dft_flux(compFields->normal_direction(flux_area), flux_area, freq_min, freq_max, Nlengths));
    FluxDFT flux_dft(meep_dft_flux);
    return flux_dft;
}

/*
Field DFT
*/

FieldsDFT FDTDSolver::addFieldDFT(const boost::python::list& field_names,
                                  plask::Vec<2, double> p1,
                                  plask::Vec<2, double> p2,
                                  double wl) {
    auto n_fields = boost::python::len(field_names);
    meep::volume flux_area(meep::vec(p1[0], p1[1]), meep::vec(p1[1], p2[1]));
    std::vector<meep::component> fields;
    double freq = {1 / (wl * 1e-3)};

    // Extract the fields from the python list
    for (long i = 0; i < n_fields; ++i) {
        fields.push_back(ComponentMap.at(boost::python::extract<std::string>(field_names[i])));
    }
    auto dft_field =
        make_shared<meep::dft_fields>(compFields->add_dft_fields(fields.data(), fields.size(), flux_area, freq, freq, 1));
    FieldsDFT result_field(dft_field, compFields);
    return result_field;
}

/*
Harminv
*/

shared_ptr<HarminvResults> FDTDSolver::doHarminv(const std::string& component,
                                                 plask::Vec<2, double> point,
                                                 double wavelength,
                                                 double dl,
                                                 double time,
                                                 const unsigned int NBands) {
    // Specifying component and the point in space, empty vector and number of bands
    meep::component c(ComponentMap.at(component));
    meep::vec p(point[0], point[1]);
    std::vector<std::vector<double>> result_vector;
    const long long int N = timeToSteps(time);

    // Results containers and temporary holder for the field value
    dcomplex field;
    std::vector<dcomplex> field_vector(N);
    shared_ptr<std::vector<dcomplex>> amp = make_shared<std::vector<dcomplex>>(NBands);
    shared_ptr<std::vector<double>> freq_re = make_shared<std::vector<double>>(NBands);
    shared_ptr<std::vector<double>> freq_im = make_shared<std::vector<double>>(NBands);
    shared_ptr<std::vector<double>> freq_err = make_shared<std::vector<double>>(NBands);

    // Calculations of the frequencies
    double cfreq = 1 / (wavelength * 1e-3);  // Central frequency
    double fmax = 1 / ((wavelength - (dl / 2)) * 1e-3);
    double fmin = 1 / ((wavelength + (dl / 2)) * 1e-3);

    // Gathering the field data for some aditional time
    writelog(LOG_INFO, "Harminv: The simulation will be performed for additional " + std::to_string(N) + " steps.");
    for (unsigned int i = 0; i < N; ++i) {
        compFields->step();
        field = compFields->get_field(c, p);
        field_vector[i] = field;
    }
    elapsedTime += (dt * baseTimeUnit * N);

    // Get the number of results
    int n_res = meep::do_harminv(field_vector.data(), N, compFields->dt, fmin, fmax, NBands, amp->data(), freq_re->data(),
                                 freq_im->data(), freq_err->data());

    shared_ptr<HarminvRawData> data = boost::make_shared<HarminvRawData>(amp, freq_re, freq_im, freq_err);
    return make_shared<HarminvResults>(data, n_res);
}

void FDTDSolver::setGeometry() {
    // Geometry initialization
    double x, y;
    x = geometry->getChildBoundingBox().width();
    y = geometry->getChildBoundingBox().height();
    meep::vec translation_vector(geometry->getChildBoundingBox().lower[0] - shiftPML,
                                 geometry->getChildBoundingBox().lower[1] - shiftPML);

    compBounds = meep::vol2d(x + 2 * shiftPML, y + 2 * shiftPML, resolution);  // X Y RES
    compBounds.set_origin(translation_vector);
    PlaskMaterialFunction matFunction(this);
    compStructure = make_shared<meep::structure>(compBounds, matFunction, meep::pml(layerPML), meep::identity(), 0, courantFactor);
    compFields = make_shared<meep::fields>(compStructure.get());  // It may be quite bad
}

void FDTDSolver::setFieldVec() {
    // Field vector initialization
    auto size = compBounds.nx() * compBounds.ny();
    plask::Vec<3, dcomplex> dummyVec({0, 0}, {0, 0}, {0, 0});
    fieldValues = DataVector<plask::Vec<3, dcomplex>>(size, dummyVec);
}

void FDTDSolver::setSource() {
    double freq = 0;
    double et = meep::infinity;
    double st = 0;

    for (auto& s : sourceContainer) {
        // First set correct frequency and times in the MEEP domain
        freq = 1 / (s.lam * 1e-3);
        if (s.end_time != "inf")
            et = fsToTimeMEEP(std::stod(s.end_time));
        else
            et = meep::infinity;
        st = fsToTimeMEEP(s.start_time);

        // Add processed sources
        if (s.source_type == "point_cont") {
            auto source_ptr = plask::make_shared<meep::continuous_src_time>(freq, 0., st, et, s.slowness);
            compFields->add_point_source(ComponentMap.at(s.component), *source_ptr, meep::vec(s.ax, s.ay), s.amplitude);
        } else if (s.source_type == "point_gauss") {
            double l_freq = (1 / ((s.lam + (s.width / 2)) * 1e-3));
            double h_freq = (1 / ((s.lam - (s.width / 2)) * 1e-3));
            auto source_ptr = plask::make_shared<meep::gaussian_src_time>(freq, h_freq - l_freq);  // s.start_time, et);
            compFields->add_point_source(ComponentMap.at(s.component), *source_ptr, meep::vec(s.ax, s.ay), s.amplitude);
        } else if (s.source_type == "vol_cont") {
            auto source_ptr = plask::make_shared<meep::continuous_src_time>(freq, 0., st, et, s.slowness);
            meep::volume pos_vec = meep::volume(meep::vec(s.ax, s.ay), meep::vec(s.bx, s.by));
            compFields->add_volume_source(ComponentMap.at(s.component), *source_ptr, pos_vec, s.amplitude);
        } else if (s.source_type == "vol_gauss") {
            double l_freq = (1 / ((s.lam + (s.width / 2)) * 1e-3));
            double h_freq = (1 / ((s.lam - (s.width / 2)) * 1e-3));
            auto source_ptr = plask::make_shared<meep::gaussian_src_time>(freq, h_freq - l_freq);  // s.start_time, et);
            meep::volume pos_vec = meep::volume(meep::vec(s.ax, s.ay), meep::vec(s.bx, s.by));
            compFields->add_volume_source(ComponentMap.at(s.component), *source_ptr, pos_vec, s.amplitude);
        }
    }
}

void FDTDSolver::setTimestep() { dt = courantFactor / resolution; }  // Sets the timestep in MEEP units

void FDTDSolver::setMesh() {
    fieldMesh->setAxis(0, plask::make_shared<RegularAxis>(compBounds.xmin(), compBounds.xmax(), compBounds.nx()));
    fieldMesh->setAxis(1, plask::make_shared<RegularAxis>(compBounds.ymin(), compBounds.ymax(), compBounds.ny()));
}

void FDTDSolver::onInitialize() {
    if (!geometry) throw NoGeometryException(getId());
    setGeometry();
    setFieldVec();
    setSource();
    setTimestep();
    setMesh();
    elapsedTime = 0;
}

LazyData<plask::Vec<3, dcomplex>> FDTDSolver::getLightH(const plask::shared_ptr<const MeshD<2>> dest_mesh,
                                                        InterpolationMethod int_method) {
    getField({"hx", "hy", "hz"});
    return interpolate(fieldMesh, fieldValues, dest_mesh, int_method);
}

LazyData<plask::Vec<3, dcomplex>> FDTDSolver::getLightE(const plask::shared_ptr<const MeshD<2>> dest_mesh,
                                                        InterpolationMethod int_method) {
    getField({"ex", "ey", "ez"});
    return interpolate(fieldMesh, fieldValues, dest_mesh, int_method);
}

long long int FDTDSolver::timeToSteps(double time) { return (int)std::ceil(time / (dt * baseTimeUnit)); }

double FDTDSolver::fsToTimeMEEP(double time) { return time / baseTimeUnit; }

}}}  // namespace plask::solvers::optical_fdtd
