import json
from collections.abc import Mapping


class StackManager:

    def __init__(self, dbg_path):
        self.ignored_vars = None
        self.stack = []
        self.dbg_path = dbg_path

    def _frame_id(self, frame):
        return id(frame)

    def _should_show_frame(self, frame):
        if frame.f_code.co_filename.endswith("bdb.py") or frame.f_code.co_filename.endswith("dbg.py"):
            return False

        return True

    def _filter_frame_locals(self, frame):

        def safe_repr(obj):
            try:
                return repr(obj)
            except Exception:
                return "<unrepresentable object>"

        def to_safe(value):
            try:
                json.dumps(value)
                return value
            except Exception:
                return {"__type__": safe_repr(type(value)), "__repr__": safe_repr(value)}

        def sanitize(obj, seen=None):
            if seen is None:
                seen = set()

            try:
                obj_id = id(obj)
                if obj_id in seen:
                    return {"__circular__": True}
                seen.add(obj_id)
            except Exception:
                return "<untrackable object>"

            # Handle dict-like safely
            try:
                if isinstance(obj, Mapping):
                    result = {}
                    for k, v in obj.items():
                        try:
                            result[k] = sanitize(v, seen)
                        except Exception:
                            result[k] = "<error>"
                    return result
            except Exception:
                pass

            # Handle list/tuple safely (avoid Iterable!)
            try:
                if isinstance(obj, (list, tuple)):
                    return [sanitize(v, seen) for v in obj]
            except Exception:
                pass

            # Fallback
            return to_safe(obj)

        if self.ignored_vars == None:
            self.ignored_vars = {}

        local_vars = {k: v for k, v in frame.f_locals.items() if k not in self.ignored_vars}

        return sanitize(local_vars)

    def _frame_info(self, frame):
        return {
            "id": self._frame_id(frame),
            "function": frame.f_code.co_name,
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
            "locals": self._filter_frame_locals(frame),
        }

    def rebuild_from_frame(self, frame, ignored_vars=None):
        new_stack = []
        f = frame
        self.ignored_vars = ignored_vars

        while f is not None:
            if self._should_show_frame(f):
                info = self._frame_info(f)
                new_stack.append(info)

            f = f.f_back

            if f is None:
                break

            if f.f_code.co_filename == self.dbg_path:
                break

        #if new_stack:
        #    new_stack.pop(-1)

        new_stack.reverse()
        self.stack = new_stack

    def get_stack(self):
        return list(self.stack)
