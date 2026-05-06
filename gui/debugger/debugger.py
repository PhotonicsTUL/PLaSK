import bdb
import queue

class Debugger(bdb.Bdb):
    def __init__(self):
        super().__init__()
        self.command_queue = queue.Queue()
        self._stop_requested = False

        # call backs 
        self.on_line = None
        self.on_paused = None
        self.on_call = None
        self.on_return = None
        self.on_exception = None
        self.on_quit = None

        self.frame = None

    def run(self, cmd, globals=None, locals=None):
        self._finished = False
        try:
            self.set_step()
            super().run(cmd, globals, locals)
        except bdb.BdbQuit:
            pass
        finally:
            self._finished = True
            if self.on_quit:
                self.on_quit()

    def stop(self):
        self._stop_requested = True
        self.set_quit()

    def user_line(self, frame):
        self.frame = frame
        if self.on_line:
            self.on_line(frame)

        if self._stop_requested:
            raise bdb.BdbQuit

        if self.on_paused:
            self.on_paused(frame)

        while not self.command_queue.empty():
            self.command_queue.get_nowait()

        self.wait_for_command()

    def user_call(self, frame, args):
        if self.on_call:
            self.on_call(frame)

    def user_return(self, frame, retval):
        if self.on_return:
            self.on_return(frame, retval)

    def user_exception(self, frame, exc_info):
        if self.on_return:
            self.on_return(frame, exc_info)
        self.set_continue()

    def wait_for_command(self):
        # blocking
        command = self.command_queue.get()
        if command == "step_into":
            self.set_step()
        elif command == "next_line":
            self.set_next(self.frame)
        elif command == "step_out":
            self.set_return(self.frame)
        elif command == "continue":
            self.set_continue()
        elif command == "quit":
            self.stop()
        else:
            print(f"[DEBUGGER]: unknown command: {command}", flush=True)

    def send_command(self, cmd):
        self.command_queue.put(cmd)
