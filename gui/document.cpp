#include "document.h"

Document::Document()
{
}

void Document::open(const QString &fileName) {
    //TODO support file names with non-asci char
    manager.loadFromFile(fileName.toStdString());
}
