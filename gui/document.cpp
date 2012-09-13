#include "document.h"
#include "material.h"


Document::Document(QtAbstractPropertyBrowser& browser): propertiesBrowser(browser) {
}

void Document::clear() {
    treeModel.clear();
    undoStack.clear();
}

void Document::save(plask::XMLWriter &dest) {
    plask::XMLWriter::Element root_object(dest, plask::Manager::TAG_NAME_ROOT);
    treeModel.save(root_object);
}

void Document::save(const std::string &filename) {
    plask::XMLWriter dest(filename);
    save(dest);
}

void Document::open(const QString &fileName) {
    plask::Manager manager;
    undoStack.clear();
    //TODO support file names with non-asci char
    manager.loadFromFile(fileName.toStdString(), &NameOnlyMaterial::getInstance);
    for (auto& object: manager.namedObjects) {
        ext(object.second)->setName(QString(object.first.c_str()));
    }
    treeModel.refresh(manager.roots);
}
