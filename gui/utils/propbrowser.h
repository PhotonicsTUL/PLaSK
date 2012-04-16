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

//factories
#include <QtDoubleSpinBoxFactory>
#include <QtSpinBoxFactory>

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

    BrowserWithManagers(QtAbstractPropertyBrowser& browser);

    /// Call clear().
    ~BrowserWithManagers() { clear(); }

    /// Clear all properties and objects in objectsToClear list.
    void clear();

    /// List of objects which are delete when clear() is called.
    QObjectList objectsToClear;

    template <class FunctorSlotType, class ReceiverT>
    bool connect(QObject *sender, const char *signal, const ReceiverT &reciever, Qt::ConnectionType type = Qt::AutoConnection) {
        QObject* conn = FunctorSlot::connect<FunctorSlotType, ReceiverT>(sender, signal, reciever, type);
        if (conn) {
            objectsToClear.append(conn);
            return true;
        } else
            return false;
    }

private:    // factories:

    QtSpinBoxFactory integerFact;
    QtDoubleSpinBoxFactory doublFact;

};

#endif // PROPBROWSER_H
