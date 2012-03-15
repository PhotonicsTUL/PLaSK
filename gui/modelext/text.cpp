#include "text.h"

#include <QObject>

#include <unordered_map>
#include <typeinfo>
#include <typeindex>

#include <plask/geometry/transform.h>
#include <plask/geometry/stack.h>
#include <plask/geometry/leaf.h>
#include <boost/lexical_cast.hpp>

typedef QString print_element_f(const plask::GeometryElement& toPrint);

std::unordered_map<std::type_index, print_element_f*> printers;

template <int dim>
QString printTranslation(const plask::GeometryElement& toPrint) {
    const plask::Translation<dim>& t = static_cast< const plask::Translation<dim>& >(toPrint);
    return QString(QObject::tr("translation%1d %2"))
            .arg(dim).arg(QString(boost::lexical_cast<std::string>(t.translation).c_str()));
}

template <int dim>
QString printStack(const plask::GeometryElement& toPrint) {
    const plask::StackContainer<dim>& t = static_cast< const plask::StackContainer<dim>& >(toPrint);
    return QString(QObject::tr("stack%1d\n%2 children"))
            .arg(dim).arg(t.getChildCount());
}

template <int dim>
QString printMultiStack(const plask::GeometryElement& toPrint) {
    const plask::MultiStackContainer<dim>& t = static_cast< const plask::MultiStackContainer<dim>& >(toPrint);
    return QString(QObject::tr("multi-stack%1d\n%2 children (%3 repeated %4 times)"))
            .arg(dim).arg(t.getChildCount()).arg(t.getRealChildCount()).arg(t.repeat_count);
}

template <int dim>
QString printBlock(const plask::GeometryElement& toPrint) {
    const plask::Block<dim>& t = static_cast< const plask::Block<dim>& >(toPrint);
    return QString(QObject::tr("block%1d\nsize: %2"))
            .arg(dim).arg(QString(boost::lexical_cast<std::string>(t.size).c_str()));
}

template <typename T>
void appendPrinter(print_element_f printer) {
    plask::shared_ptr<T> o = plask::make_shared<T>();
    printers[std::type_index(typeid(*o))] = printer;
}

void initElementsPrinters() {
    appendPrinter<plask::Translation<2>>(printTranslation<2>);
    appendPrinter<plask::Translation<3>>(printTranslation<3>);
    appendPrinter<plask::MultiStackContainer<2>>(printMultiStack<2>);
    appendPrinter<plask::MultiStackContainer<3>>(printMultiStack<3>);
    appendPrinter<plask::StackContainer<2>>(printStack<2>);
    appendPrinter<plask::StackContainer<3>>(printStack<3>);
    appendPrinter<plask::Block<2>>(printBlock<2>);
    appendPrinter<plask::Block<3>>(printBlock<3>);
}

QString toStr(plask::GeometryElement::Type type) {
    switch (type) {
    case plask::GeometryElement::TYPE_CONTAINER: return QObject::tr("container");
    case plask::GeometryElement::TYPE_LEAF: return QObject::tr("leaf");
    case plask::GeometryElement::TYPE_SPACE_CHANGER: return QObject::tr("space changer");
    case plask::GeometryElement::TYPE_TRANSFORM: return QObject::tr("transform");
    }
}

QString universalPrinter(const plask::GeometryElement& el) {
    return QString(QObject::tr("%1%2d\n%3 children")
            .arg(toStr(el.getType())))
            .arg(el.getDimensionsCount())
            .arg(el.getChildCount());
}

QString toStr(const plask::GeometryElement& el) {
    auto printer = printers.find(std::type_index(typeid(el)));
    return printer != printers.end() ?
        printer->second(el) :
        universalPrinter(el);
}
