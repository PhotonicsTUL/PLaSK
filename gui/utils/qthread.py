# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from ..qt import QtCore


class BackgroundTask(QtCore.QThread):

    _tasks = set()

    def __init__(self, task, callback=None):
        super(BackgroundTask, self).__init__()
        self._task = task
        self._callback = callback
        BackgroundTask._tasks.add(self)
        self.finished.connect(self._task_finished)
        self._result = None

    def run(self):
        self._result = self._task()

    def _task_finished(self):
        try:
            BackgroundTask._tasks.remove(self)
        except KeyError:
            pass
        if self._callback is not None:
            if self._result is None:
                self._callback()
            elif type(self._result) is tuple:
                self._callback(*self._result)
            else:
                self._callback(self._result)


class Lock(object):

    def __init__(self, mutex, blocking=True):
        self._mutex = mutex
        self._blocking = blocking
        self._locked = False

    def __enter__(self):
        if self._blocking:
            self._mutex.lock()
            self._locked = True
        else:
            self._locked = self._mutex.tryLock()
        return self._locked

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._locked:
            self._mutex.unlock()
