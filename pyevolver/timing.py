from pytictoc import TicToc
from collections import defaultdict


class Timing:

    def __init__(self, active):
        self.active = active
        if active:
            self.elapsed = defaultdict(float)

    def init_ellapsed(self):
        if self.active:
            self.elapsed = defaultdict(float)

    def init_tictoc(self):
        if self.active:
            t = TicToc()
            t.tic()
            return t
        return None

    def add_time(self, field_name, t):
        if self.active:
            self.elapsed[field_name] += t.tocvalue(restart=True)

    def report(self):

        if not self.active:
            return

        cats = sorted(set(x.split('_')[0] for x in self.elapsed.keys()))
        max_str_len = max(len(k) for k in self.elapsed)

        for c in cats:
            total = sum(v for k, v in self.elapsed.items() if k.startswith(c))
            print("\n{} total time: {:.2f}".format(c, total))
            cat_elapsed = {k: v for k, v in self.elapsed.items() if k.startswith(c)}
            for k, v in sorted(cat_elapsed.items(), key=lambda kv: -kv[1]):
                t = '{:.2f}'.format(v)
                p = '{:.2f}%'.format(v / total * 100)
                print("{}  {}  {}".format(k.ljust(max_str_len), t.rjust(7), p.rjust(7)))
