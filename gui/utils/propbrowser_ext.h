#ifndef PLASK_GUI_UTILS_PROPBROWSER_ALIGNER_H
#define PLASK_GUI_UTILS_PROPBROWSER_ALIGNER_H

#include <QtLineEditFactory>
#include <QCompleter>

class QtLineEditWithCompleterFactory: public QtLineEditFactory
{
    QCompleter* completer;

public:
    QtLineEditWithCompleterFactory(QObject *parent = 0)
        : QtLineEditFactory(parent), completer(new QCompleter(this)) {
        completer->setCompletionMode(QCompleter::UnfilteredPopupCompletion);
    }

    QtLineEditWithCompleterFactory(const QStringList& wordList, QObject *parent = 0)
        : QtLineEditFactory(parent), completer(new QCompleter(wordList, this)) {
        completer->setCompletionMode(QCompleter::UnfilteredPopupCompletion);
    }

    QWidget *createEditor(QtStringPropertyManager *manager, QtProperty *property, QWidget *parent);

    QCompleter& getCompleter() { return *completer; }
    const QCompleter& getCompleter() const { return *completer; }
};

#endif // PLASK_GUI_UTILS_PROPBROWSER_ALIGNER_H
