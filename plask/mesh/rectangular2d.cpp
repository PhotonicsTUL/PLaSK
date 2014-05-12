#include "rectangular2d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

namespace plask {

static std::size_t normal_index(const RectangularMesh<2>* mesh, std::size_t index0, std::size_t index1) {
    return index0 + mesh->axis0->size() * index1;
}
static std::size_t normal_index0(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis0->size();
}
static std::size_t normal_index1(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis0->size();
}

static std::size_t transposed_index(const RectangularMesh<2>* mesh, std::size_t index0, std::size_t index1) {
    return mesh->axis1->size() * index0 + index1;
}
static std::size_t transposed_index0(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis1->size();
}
static std::size_t transposed_index1(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis1->size();
}

void RectangularMesh<2>::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == ORDER_01) {
        index_f = transposed_index;
        index0_f = transposed_index0;
        index1_f = transposed_index1;
        minor_axis = &axis1;
        major_axis = &axis0;
    } else {
        index_f = normal_index;
        index0_f = normal_index0;
        index1_f = normal_index1;
        minor_axis = &axis0;
        major_axis = &axis1;
    }
    this->fireChanged();
}

typename RectangularMesh<2>::IterationOrder RectangularMesh<2>::getIterationOrder() const {
    return (index_f == &transposed_index)? ORDER_01 : ORDER_10;
}

void RectangularMesh<2>::setAxis(const shared_ptr<RectangularAxis> &axis, shared_ptr<RectangularAxis> new_val) {
    if (axis == new_val) return;
    unsetChangeSignal(axis);
    const_cast<shared_ptr<RectangularAxis>&>(axis) = new_val;
    setChangeSignal(axis);
    fireResized();
}

void RectangularMesh<2>::onAxisChanged(Mesh::Event &e) {
    assert(!e.isDelete());
    this->fireChanged(e.flags());
}

RectangularMesh<2>::~RectangularMesh() {
    unsetChangeSignal(axis0);
    unsetChangeSignal(axis1);
}

shared_ptr<RectangularMesh<2> > RectangularMesh<2>::getMidpointsMesh() {
    return make_shared<RectangularMesh<2>>(axis0->getMidpointsMesh(), axis1->getMidpointsMesh(), getIterationOrder());
}

void RectangularMesh<2>::writeXML(XMLElement& object) const {
    object.attr("type", "rectangular2d");
    { auto a = object.addTag("axis0"); axis0->writeXML(a); }
    { auto a = object.addTag("axis1"); axis1->writeXML(a); }
}


// Particular instantations
template class RectangularMesh<2>;

static shared_ptr<Mesh> readRectangularMesh2D(XMLReader& reader) {
    shared_ptr<RectangularAxis> axis[2];
    XMLReader::CheckTagDuplication dub_check;
    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();
        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
        dub_check(std::string("<mesh>"), node);
        boost::optional<std::string> type = reader.getAttribute("type");
        if (type) {
            if (*type == "regular") axis[node[4]-'0'] = readRegularMesh1D(reader);
            else if (*type == "rectilinear") axis[node[4]-'0'] = readRectilinearMesh1D(reader);
            else throw XMLBadAttrException(reader, "type", *type, "\"regular\" or \"rectilinear\"");
        } else {
            if (reader.hasAttribute("start")) axis[node[4]-'0'] = readRegularMesh1D(reader);
            else axis[node[4]-'0'] = readRectilinearMesh1D(reader);
        }
    }
    reader.requireTagEnd();
    return make_shared<RectangularMesh<2>>(std::move(axis[0]), std::move(axis[1]));
}

static RegisterMeshReader rectangular2d_reader("rectangular2d", readRectangularMesh2D);

// deprecated:
static RegisterMeshReader regularmesh2d_reader("regular2d", readRectangularMesh2D);
static RegisterMeshReader rectilinear2d_reader("rectilinear2d", readRectangularMesh2D);


/*
static shared_ptr<Mesh> readRegularMesh2D(XMLReader& reader)
{
    std::map<std::string,std::tuple<double,double,size_t>> axes;

    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), node);

        double start = reader.requireAttribute<double>("start");
        double stop = reader.requireAttribute<double>("stop");
        size_t count = reader.requireAttribute<size_t>("num");
        axes[node] = std::make_tuple(start, stop, count);

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RegularMesh2D>(RegularAxis(std::get<0>(axes["axis0"]), std::get<1>(axes["axis0"]), std::get<2>(axes["axis0"])),
                                      RegularAxis(std::get<0>(axes["axis1"]), std::get<1>(axes["axis1"]), std::get<2>(axes["axis1"])));
}

static shared_ptr<Mesh> readRegularMesh3D(XMLReader& reader)
{
    std::map<std::string,std::tuple<double,double,size_t>> axes;

    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), "tag <" + node + ">");

        double start = reader.requireAttribute<double>("start");
        double stop = reader.requireAttribute<double>("stop");
        size_t count = reader.requireAttribute<size_t>("num");
        axes[node] = std::make_tuple(start, stop, count);

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RegularMesh3D>(RegularAxis(std::get<0>(axes["axis0"]), std::get<1>(axes["axis0"]), std::get<2>(axes["axis0"])),
                                      RegularAxis(std::get<0>(axes["axis1"]), std::get<1>(axes["axis1"]), std::get<2>(axes["axis1"])),
                                      RegularAxis(std::get<0>(axes["axis2"]), std::get<1>(axes["axis2"]), std::get<2>(axes["axis2"])));
}

static RegisterMeshReader regularmesh2d_reader("regular2d", readRegularMesh2D);
static RegisterMeshReader regularmesh3d_reader("regular3d", readRegularMesh3D);


template<>
shared_ptr<RectilinearMesh2D> RectilinearMesh2D::getMidpointsMesh() {

    if (this->midpoints_cache) return this->midpoints_cache;

    if (this->axis0.size() < 2 || this->axis1.size() < 2)
        throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearAxis line0;
    for (auto a = this->axis0.begin(), b = this->axis0.begin()+1; b != this->axis0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearAxis line1;
    for (auto a = this->axis1.begin(), b = this->axis1.begin()+1; b != this->axis1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    this->midpoints_cache = make_shared<RectilinearMesh2D>(line0, line1, this->getIterationOrder());
    return this->midpoints_cache;
}

template<>
shared_ptr<RectilinearMesh3D> RectilinearMesh3D::getMidpointsMesh() {

    if (this->midpoints_cache) return this->midpoints_cache;

    if (this->axis0.size() < 2 || this->axis1.size() < 2 || this->axis2.size() < 2)
        throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearAxis line0;
    for (auto a = this->axis0.begin(), b = this->axis0.begin()+1; b != this->axis0.end(); ++a, ++b)
        line0.addPoint((*a + *b) / 2.0);

    RectilinearAxis line1;
    for (auto a = this->axis1.begin(), b = this->axis1.begin()+1; b != this->axis1.end(); ++a, ++b)
        line1.addPoint((*a + *b) / 2.0);

    RectilinearAxis line2;
    for (auto a = this->axis2.begin(), b = this->axis2.begin()+1; b != this->axis2.end(); ++a, ++b)
        line2.addPoint((*a + *b) / 2.0);

    this->midpoints_cache = make_shared<RectilinearMesh3D>(line0, line1, line2, this->getIterationOrder());
    return this->midpoints_cache;
}

template <>
void RectilinearMesh2D::writeXML(XMLElement& object) const {
    object.attr("type", "rectilinear2d");
    for (size_t n = 0; n != 2; ++n) {
        auto axis = object.addTag("axis"+boost::lexical_cast<std::string>(n));
        axis.indent();
        for (auto x: this->axis(n)) {
            axis.writeText(x);
            axis.writeText(" ");
        }
        axis.writeText("\n");
    }
}

template <>
void RectilinearMesh3D::writeXML(XMLElement& object) const {
    object.attr("type", "rectilinear3d");
    for (size_t n = 0; n != 3; ++n) {
        auto axis = object.addTag("axis"+boost::lexical_cast<std::string>(n));
        axis.indent();
        for (auto x: this->axis(n)) {
            axis.writeText(x);
            axis.writeText(" ");
        }
        axis.writeText("\n");
    }
}



static shared_ptr<Mesh> readRectilinearMesh2D(XMLReader& reader)
{
    std::map<std::string,RectilinearAxis> axes;

    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), "tag <" + node + ">");

        if (reader.hasAttribute("start")) {
            double start = reader.requireAttribute<double>("start");
            double stop = reader.requireAttribute<double>("stop");
            size_t count = reader.requireAttribute<size_t>("num");
            axes[node].addPointsLinear(start, stop, count);
            reader.requireTagEnd();
        } else {
            std::string data = reader.requireTextInCurrentTag();
            for (auto point: boost::tokenizer<>(data)) {
                try {
                    double val = boost::lexical_cast<double>(point);
                    axes[node].addPoint(val);
                } catch (boost::bad_lexical_cast) {
                    throw XMLException(reader, format("Value '%1%' cannot be converted to float", point));
                }
            }
        }
    }
    reader.requireTagEnd();

    return make_shared<RectilinearMesh2D>(std::move(axes["axis0"]), std::move(axes["axis1"]));
}

static shared_ptr<Mesh> readRectilinearMesh3D(XMLReader& reader)
{
    std::map<std::string,RectilinearAxis> axes;

    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1" && node != "axis2") throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), "tag <" + node + ">");

       if (reader.hasAttribute("start")) {
            double start = reader.requireAttribute<double>("start");
            double stop = reader.requireAttribute<double>("stop");
            size_t count = reader.requireAttribute<size_t>("num");
            axes[node].addPointsLinear(start, stop, count);
            reader.requireTagEnd();
        } else {
            std::string data = reader.requireTextInCurrentTag();
            for (auto point: boost::tokenizer<>(data)) {
                try {
                    double val = boost::lexical_cast<double>(point);
                    axes[node].addPoint(val);
                } catch (boost::bad_lexical_cast) {
                    throw XMLException(reader, format("Value '%1%' cannot be converted to float", point));
                }
            }
        }
    }
    reader.requireTagEnd();

    return make_shared<RectilinearMesh3D>(std::move(axes["axis0"]), std::move(axes["axis1"]), std::move(axes["axis2"]));
}

static RegisterMeshReader rectilinearmesh2d_reader("rectilinear2d", readRectilinearMesh2D);
static RegisterMeshReader rectilinearmesh3d_reader("rectilinear3d", readRectilinearMesh3D);

*/


shared_ptr<RectangularMesh<2> > make_rectilinear_mesh(const RectangularMesh<2> &to_copy) {
   return make_shared<RectangularMesh<2>>(make_shared<RectilinearAxis>(*to_copy.axis0), make_shared<RectilinearAxis>(*to_copy.axis1), to_copy.getIterationOrder());
}

} // namespace plask



