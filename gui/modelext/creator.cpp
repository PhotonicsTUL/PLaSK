#include "creator.h"

#include <plask/geometry/leaf.h>
#include <plask/geometry/stack.h>

struct BlockCreator: public GeometryElementCreator {

    virtual plask::shared_ptr<plask::GeometryElement> getElement(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::Block<2> >(plask::vec(1.0, 1.0));
            case 3: return plask::make_shared< plask::Block<3> >(plask::vec(1.0, 1.0, 1.0));
        }
        return plask::shared_ptr<plask::GeometryElement>();
    }

    virtual std::string getName() const { return "block"; }

};

struct StackCreator: public GeometryElementCreator {

    virtual plask::shared_ptr<plask::GeometryElement> getElement(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::StackContainer<2> >();
            case 3: return plask::make_shared< plask::StackContainer<3> >();
        }
        return plask::shared_ptr<plask::GeometryElement>();
    }

    virtual std::string getName() const { return "stack"; }

};

struct MultiStackCreator: public GeometryElementCreator {

    virtual plask::shared_ptr<plask::GeometryElement> getElement(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::MultiStackContainer<2> >();
            case 3: return plask::make_shared< plask::MultiStackContainer<3> >();
        }
        return plask::shared_ptr<plask::GeometryElement>();
    }

    virtual std::string getName() const { return "multistack"; }

};

struct ShelfCreator: public GeometryElementCreator {

    virtual plask::shared_ptr<plask::GeometryElement> getElement(int dim) const {
        switch (dim) {
            case 2: return plask::make_shared< plask::ShelfContainer2D >();
        }
        return plask::shared_ptr<plask::GeometryElement>();
    }

    virtual std::string getName() const { return "shelf2D"; }

    virtual bool supportDimensionsCount(int dim) const {
        return dim == 2;
    }

};

const std::vector<const GeometryElementCreator*>& getCreators() {
    static std::vector<const GeometryElementCreator*> vec = { new BlockCreator(), new StackCreator(), new MultiStackCreator(), new ShelfCreator() };
    return vec;
}

const std::vector<const GeometryElementCreator*>& getCreators(int dim) {
    static std::vector<const GeometryElementCreator*> vec2 = { new BlockCreator(), new StackCreator(), new MultiStackCreator(), new ShelfCreator() };
    static std::vector<const GeometryElementCreator*> vec3 = { new BlockCreator(), new StackCreator(), new MultiStackCreator() };
    static std::vector<const GeometryElementCreator*> empty_vec;
    switch (dim) {
        case 2: return vec2;
        case 3: return vec3;
    }
    return empty_vec;
}

GeometryElementCreator* fromMimeData(const QMimeData * data) {
    QByteArray ptrData = data->data(MIME_PTR_TO_CREATOR);
    QDataStream stream(&ptrData, QIODevice::ReadOnly);

    GeometryElementCreator* creator = 0;
    stream.readRawData(reinterpret_cast<char*>(&creator), sizeof(creator));
    return creator;
}
