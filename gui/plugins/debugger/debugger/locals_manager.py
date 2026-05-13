import inspect

class LocalsManager:
    def __init__(self):
        self.local_variables = {}

    def update_locals(self, frame):
        if not self._should_show_frame(frame):
            self.local_variables = {}
            return

        self.local_variables = self._filter_frame_locals(frame)

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

    def _filter_frame_locals(self, frame):
        return {
            k: v for k, v in frame.f_locals.items()
            if not k.startswith("__")
        }

    def get_locals(self):
        return dict(self.local_variables)
