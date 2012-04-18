#ifndef PLASK_GUI_UTILS_PROPBROWSER_H
#define PLASK_GUI_UTILS_PROPBROWSER_H

/** @file
 * This file includes utils functions and classes connected with QT property browser.
 */

#include <QtAbstractPropertyBrowser>

//managers
#include <QtDoublePropertyManager>
#include <QtSizeFPropertyManager>
#include <QtIntPropertyManager>
#include <QtStringPropertyManager>

//factories
#include <QtDoubleSpinBoxFactory>
#include <QtSpinBoxFactory>
#include <QtLineEditFactory>

#include "propbrowser_ext.h"

#include "slots.h"

/**
 * Includes property browser and set of property managers for it.
 */
//TODO browser should listen on browsed object changes, it can be even deleted!
struct BrowserWithManagers {

    QtAbstractPropertyBrowser& browser;

    // managers:
    //TODO read-only versions (without editor), if needed
    QtIntPropertyManager integer;
    QtDoublePropertyManager doubl;
    QtSizeFPropertyManager sizeF;
    QtStringPropertyManager string;
    QtStringPropertyManager aligner;

    BrowserWithManagers(QtAbstractPropertyBrowser& browser);

    /// Call clear().
    ~BrowserWithManagers() { clear(); }

    /// Clear all properties and objects in objectsToClear list.
    void clear();

    /// List of objects which are delete when clear() is called.
    QObjectList objectsToClear;

    template <class FunctorSlotType, class ReceiverT>
    bool connect(QObject *sender, const char *signal, const ReceiverT &receiver, Qt::ConnectionType type = Qt::AutoConnection) {
        QObject* conn = FunctorSlot::connect<FunctorSlotType, ReceiverT>(sender, signal, receiver, type);
        if (conn) {
            objectsToClear.append(conn);
            return true;
        } else
            return false;
    }

    template <class ReceiverT>
    bool connectInt(QtProperty* property, const ReceiverT &receiver, Qt::ConnectionType type = Qt::AutoConnection) {
        return connect<FunctorSlot::PropertyInteger>(property->propertyManager(), SIGNAL(valueChanged(QtProperty*, int)),
                       [=](QtProperty* p, int v) { if (p == property) receiver(v); }, type);
    }

    template <class ReceiverT>
    bool connectDouble(QtProperty* property, const ReceiverT &receiver, Qt::ConnectionType type = Qt::AutoConnection) {
        return connect<FunctorSlot::PropertyDouble>(property->propertyManager(), SIGNAL(valueChanged(QtProperty*, double)),
                       [=](QtProperty* p, double v) { if (p == property) receiver(v); }, type);
    }

    template <class ReceiverT>
    bool connectSizeF(QtProperty* property, const ReceiverT &receiver, Qt::ConnectionType type = Qt::AutoConnection) {
        return connect<FunctorSlot::PropertyQSizeF>(property->propertyManager(), SIGNAL(valueChanged(QtProperty*, const QSizeF &)),
                       [=](QtProperty* p, const QSizeF &v) { if (p == property) receiver(v); }, type);
    }

    template <class ReceiverT>
    bool connectString(QtProperty* property, const ReceiverT &receiver, Qt::ConnectionType type = Qt::AutoConnection) {
        return connect<FunctorSlot::PropertyQSizeF>(property->propertyManager(), SIGNAL(valueChanged(QtProperty*, const QString &)),
                       [=](QtProperty* p, const QString &v) { if (p == property) receiver(v); }, type);
    }

private:    // factories:

    QtSpinBoxFactory integerFact;
    QtDoubleSpinBoxFactory doublFact;
    QtLineEditFactory stringFact;

    QtLineEditWithCompleterFactory alignerFact;

};

#endif // PROPBROWSER_H
