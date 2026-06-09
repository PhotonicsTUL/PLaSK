
class WatchlistManager:
    def __init__(self) -> None:
        self.expressions = []
        self.watchlist = {}

    def eval_watchlist(self, frame):
        for e in self.expressions:
            self.watchlist[e] = self._evaluate_expression(e, frame)

    def _evaluate_expression(self, expr: str, frame):
        try:
            return eval(expr, frame.f_globals, frame.f_locals)
        except Exception as e:
            return f"<Error: {e.__class__.__name__}: {e}>"

    def update_watchlist(self, expressions):
        self.expressions = expressions

    def get_watchlist(self):
        return self.watchlist

