#ifndef PLASK_GUI_UTILS_SLOTS_H
#define PLASK_GUI_UTILS_SLOTS_H

/** @file
 * This file includes utils functions and classes connected with QT signals and slots.
 */

#include <QObject>
#include <QtProperty>

#include <functional>

namespace FunctorSlot {

class Void: public QObject {
    Q_OBJECT
    public:
        Void(QObject *parent, const std::function<void()> &f) : QObject(parent), function_(f) {}
        const char* slotName() { return SLOT(signaled()); }
    public Q_SLOTS: void signaled() { function_(); }
    private: std::function<void()> function_;
};

class Property: public QObject {
    Q_OBJECT
    public:
        Property(QObject *parent, const std::function<void(QtProperty*)> &f) : QObject(parent), function_(f) {}
        const char* slotName() { return SLOT(signaled(QtProperty*)); }
    public Q_SLOTS: void signaled(QtProperty* p) { function_(p); }
    private: std::function<void(QtProperty*)> function_;
};

class PropertyQSizeF: public QObject {
    Q_OBJECT
    public:
        PropertyQSizeF(QObject *parent, const std::function<void(QtProperty*, const QSizeF&)> &f) : QObject(parent), function_(f) {}
        const char* slotName() { return SLOT(signaled(QtProperty*, const QSizeF&)); }
    public Q_SLOTS: void signaled(QtProperty* p, const QSizeF& s) { function_(p, s); }
    private: std::function<void(QtProperty*, const QSizeF&)> function_;
};

template <class FunctorSlotType, class ReceiverT>
FunctorSlotType* connect(QObject *sender, const char *signal, const ReceiverT &reciever, Qt::ConnectionType type = Qt::AutoConnection) {
    FunctorSlotType* s = new FunctorSlotType(sender, reciever);
    if (QObject::connect(sender, signal, s, s->slotName(), type)) {
        return s;
    } else {
        delete s;
        return 0;
    }
}

}   // SlotToFunctor



#endif // PLASK_GUI_UTILS_SLOTS_H
