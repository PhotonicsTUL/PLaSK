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

from ..qt.QtCore import *


class BackgroundTask(object):

    instances = set()

    class Thread(QThread):

        def __init__(self, task):
            super(BackgroundTask.Thread, self).__init__()
            self._task = task

        def run(self):
            self._task.result = self._task.task()

    def __init__(self, task, callback=None):
        self.task = task
        self.callback = callback
        BackgroundTask.instances.add(self)
        self.thread = BackgroundTask.Thread(self)
        self.thread.finished.connect(self.task_finished)
        self.result = None

    def start(self):
        self.thread.start()

    def task_finished(self):
        del self.thread
        if self.callback is not None:
            if type(self.result) is tuple:
                self.callback(*self.result)
            else:
                self.callback(self.result)
        try:
            BackgroundTask.instances.remove(self)
        except KeyError:
            pass


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
