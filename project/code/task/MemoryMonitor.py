import time
import os
import psutil
import threading

class MemoryMonitor(threading.Thread):
    def __init__(self, interval=0.001):
        super().__init__()
        self.interval = interval
        self.keep_running = True
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())
        self.start_memory = 0

    def run(self):
        self.start_memory = self.process.memory_info().rss
        self.peak_memory = self.start_memory
        
        while self.keep_running:
            try:
                current = self.process.memory_info().rss
                if current > self.peak_memory:
                    self.peak_memory = current
                time.sleep(self.interval)
            except:
                break

    def stop(self):
        self.keep_running = False
        
    def get_peak_delta_mb(self):
        return (self.peak_memory - self.start_memory) / (1024 * 1024)