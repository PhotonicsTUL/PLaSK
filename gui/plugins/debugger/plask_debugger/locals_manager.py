import json
from collections.abc import Iterable, Mapping


class LocalsManager:

    def __init__(self):
        self.ignored_vars = None
        self.local_variables = {}

    def update_locals(self, frame, ignored_vars=None):
        self.ignored_vars = ignored_vars
        self.local_variables = self._filter_frame_locals(frame)

    def _filter_frame_locals(self, frame):

        def to_safe(value):
            try:
                json.dumps(value)
                return value
            except (TypeError, OverflowError):
                return {"__type__": type(value).__name__, "__repr__": repr(value)}

        def sanitize(obj):
            if isinstance(obj, Mapping):
                return {k: sanitize(v) for k, v in obj.items()}

            if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                return [sanitize(v) for v in obj]

            return to_safe(obj)

        if self.ignored_vars == None:
            self.ignored_vars = {}

        local_vars = {k: v for k, v in frame.f_locals.items() if k not in self.ignored_vars}

        return sanitize(local_vars)

    def get_locals(self):
        return dict(self.local_variables)
