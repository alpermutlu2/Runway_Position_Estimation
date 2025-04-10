# utils/profiler.py

import time

class Timer:
    def __init__(self):
        self.start_times = {}
        self.elapsed = {}

    def start(self, name):
        self.start_times[name] = time.time()

    def stop(self, name):
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.elapsed[name] = self.elapsed.get(name, 0) + duration

    def report(self):
        print("\n⏱️  Timing Summary (seconds):")
        for name, duration in self.elapsed.items():
            print(f" - {name}: {duration:.3f}s")
