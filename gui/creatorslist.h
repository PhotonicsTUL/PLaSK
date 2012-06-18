#ifndef PLASK_GUI_CREATORSLIST_H
#define PLASK_GUI_CREATORSLIST_H

#include <QAbstractListModel>

QT_BEGIN_NAMESPACE
class QAbstractListModel;
QT_END_NAMESPACE

struct CreatorsListModel: public QAbstractListModel {

    int rowCount(const QModelIndex &parent) const;

    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;

};

#endif // PLASK_GUI_CREATORSLIST_H
