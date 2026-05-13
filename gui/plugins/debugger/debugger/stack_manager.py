class StackManager:
    def __init__(self, dbg_path):
        self.stack = []
        self.dbg_path = dbg_path

    def _frame_id(self, frame):
        return id(frame)

    def _should_show_frame(self, frame):
        if frame.f_code.co_name == "<module>":
            return False

        filename = frame.f_code.co_filename

        if (
            "site-packages" in filename or
            "/usr/lib/" in filename or
            "plask" in filename
        ):
            return False

        return True

    def _filter_locals(self, frame):
        return {
            k: v for k, v in frame.f_locals.items()
            if not k.startswith("__")
        }

    def _frame_info(self, frame):
        return {
            "id": self._frame_id(frame),
            "function": frame.f_code.co_name,
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
            "locals": self._filter_locals(frame),
        }

    def rebuild_from_frame(self, frame):
        new_stack = []
        f = frame

        while f is not None:
            if self._should_show_frame(f):
                info = self._frame_info(f)
                new_stack.append(info)

            f = f.f_back

            if f is None:
                break

            if f.f_code.co_filename == self.dbg_path:
                break

        if new_stack:
            new_stack.pop(-1)

        new_stack.reverse()
        self.stack = new_stack

    def get_stack(self):
        return list(self.stack)
