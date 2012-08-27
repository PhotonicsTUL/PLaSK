#include "document.h"
#include "material.h"


Document::Document(QtAbstractPropertyBrowser& browser): propertiesBrowser(browser) {
}

void Document::clear() {
    treeModel.clear();
    undoStack.clear();
}

void Document::save(plask::XMLWriter &dest) {
    plask::XMLWriter::Element root_element(dest, plask::Manager::TAG_NAME_ROOT);
    treeModel.save(root_element);
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
    for (auto& element: manager.namedElements) {
        ext(element.second)->setName(QString(element.first.c_str()));
    }
    treeModel.refresh(manager.roots);
}
