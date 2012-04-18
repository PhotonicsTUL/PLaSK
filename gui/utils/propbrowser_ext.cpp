#include "propbrowser_ext.h"

QWidget *QtLineEditWithCompleterFactory::createEditor(QtStringPropertyManager *manager, QtProperty *property, QWidget *parent) {
    QWidget *result = QtLineEditFactory::createEditor(manager, property, parent);
    if (result) {
        QLineEdit* ed = static_cast<QLineEdit*>(result);
        ed->setCompleter(completer);
    }
    return result;
}
