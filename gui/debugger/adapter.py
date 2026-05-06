import json
from debugger import Debugger
from stack_manager import StackManager
from locals_manager import LocalsManager
from watchlist_manager import WatchlistManager

import queue

# PROTOCOL 
#
# ui → debugger
# {
#   type: command
#   name: <contiue | step_line | step_into | step_out | quit | update_watchlist>
#   payload: {} 
# }
#
# debugger → ui
# {
#   type: state_update
#   name: 'state_update'
#   payload: {
#     'line': ...
#     'locals': ...
#     'call_stack': ...
#     'watch_list': ...
#   }
# }

class DebuggerAdapter:

    def __init__(self, line_offset=0) -> None:
        self.debugger = Debugger()

        self.event_queue = queue.Queue()

        self.locals_manager = LocalsManager()
        self.stack_manager = StackManager(__file__)
        self.watchlist_manager = WatchlistManager()

        self.debugger.on_line = self.handle_line
        self.debugger.on_paused = self.handle_paused
        self.debugger.on_call = self.handle_call
        self.debugger.on_return = self.handle_return 
        self.debugger.on_exception = self.handle_exception
        self.debugger.on_quit = self.handle_quit

        self.emit_state = None

        self.line_offset = line_offset
        self.current_line = -1

    def run(self, code):
        self.debugger.run(code)

    def handle_command(self, cmd):
        payload = cmd['payload']
        if cmd['name'] == 'continue':
            self.resume()
        elif cmd['name'] == 'step_line':
            self.step_line()
        elif cmd['name'] == 'step_into':
            self.step_into()
        elif cmd['name'] == 'step_out':
            self.step_out()
        elif cmd['name'] == 'update_watchlist':
            self.update_watchlist(payload)
        elif cmd['name'] == 'quit':
            self.quit()
        else:
            print(f"[Debugger]: unknown command: {cmd}", flush=True)

    def handle_quit(self):
        state = {
                    'type': 'state_update',
                    'name': 'quit',
                    'payload': {}
                }
        if self.emit_state:
            self.emit_state(json.dumps(state))

    def get_state(self):
        payload = {
                    'line': self.current_line,
                    'locals': self.locals_manager.get_locals(),
                    'call_stack': self.stack_manager.get_stack(),
                    'watch_list': self.watchlist_manager.get_watchlist(),
                }
        state = {
                    'type': 'state_update',
                    'name': 'state_update',
                    'payload': payload
                }
        try:
            json_state = json.dumps(state)
        except Exception as e:
            print(f"[DEBUGGER]: error when serialising state: {state}, error: {e}")
            json_state = "{}"
        return json_state
    
    def update_watchlist(self, watchlist):
        self.watchlist_manager.update_watchlist(list(watchlist))
    
    def handle_line(self, frame):
        self.current_line = frame.f_lineno - self.line_offset
        self.locals_manager.update_locals(frame)
        self.stack_manager.rebuild_from_frame(frame)
        self.watchlist_manager.eval_watchlist(frame)

    def handle_paused(self, frame):
        state = self.get_state()
        if self.emit_state:
            self.emit_state(state)

    def handle_call(self, frame):
        pass

    def handle_return(self, frame, retval):
        pass

    def handle_exception(self, frame, exc_info):
        pass

    def resume(self):
        self.debugger.send_command("continue")

    def step_line(self):
        self.debugger.send_command("next_line")

    def step_into(self):
        self.debugger.send_command("step_into") 

    def step_out(self):
        self.debugger.send_command("step_out")

    def quit(self):
        self.debugger.send_command("quit")
