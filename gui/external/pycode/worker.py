# -*- coding: utf-8 -*-
import sys
import os
import select
import time
import atexit
from multiprocessing import Process, Queue
from threading import Thread

parent_pid = os.getpid()


class WorkerProcess(Process):

    """
    class spawns a sub-process
    communication goes over three queues
     * input queue: to accept requests
     * output queue: to send ressponses
     * control queue: to kill process

    select is used to detect events
    """

    def __init__(self):
        super(WorkerProcess, self).__init__()
        self.inq = Queue()
        self.outq = Queue()
        self.ctrlq = Queue()
        self.counter = 0
        self.functions = {}
        self.callbacks = {}
        self.killed = False
        atexit.register(self.kill)
        self.result_listener_thread = Thread(target=self.listen_results)
        self.result_listener_thread.start()

    def register(self, function, name=None):
        self.functions[name or function.__name__] = function

    def listen_results(self, timeout=2):
        while not self.killed:
            (out_reader, _, _) = select.select(
                [self.outq._reader], [], [], timeout)
            try:
                job_id, error, result = self.outq.get(block=False)
            except:  # Empty?
                continue
            try:
                if error:
                    print "*", error
                    raise error
                else:
                    self.callbacks[job_id](result)
            except Exception, exc:
                print >>sys.stderr, "Error executing traceback", exc
                import traceback
                traceback.print_exc()

    def is_busy(self):
        return self.inq.qsize() > 0

    def run(self):
        pid = os.getpid()
        while 1:
            (inq, outq, ctrlq) = select.select(
                [self.inq._reader, self.ctrlq._reader], [], [])
            if inq and self.ctrlq._reader in inq:
                return
            job = self.inq.get()
            job_id, function, args, kwargs = job
            try:
                if callable(function):
                    result = function(*args, **kwargs)
                else:
                    result = self.functions[function](*args, **kwargs)
                error = None
            except Exception, exc:
                result = None
                error = exc
                print >>sys.stderr, "[Worker] Error executing job %s: %s" % (
                    function, error)
                import traceback
                traceback.print_exc()
            if job_id != 0:
                self.outq.put((job_id, error, result))

    def execute(self, function, *args, **kwargs):
        callback = kwargs.pop("callback", None)
        if callback is None:
            job_id = 0
        else:
            self.counter += 1
            job_id = self.counter
            self.callbacks[job_id] = callback
        job = (job_id, function, args, kwargs)
        self.inq.put(job)

    def kill(self):
        self.killed = True
        self.ctrlq.put(None)
        time.sleep(0.1)
        self.terminate()
