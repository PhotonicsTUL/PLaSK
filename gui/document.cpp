#include "document.h"

Document::Document(): treeModel(*this) {
}

void Document::open(const QString &fileName) {
    //TODO support file names with non-asci char
    manager.loadFromFile(fileName.toStdString());
    treeModel.refresh(*this);
}
