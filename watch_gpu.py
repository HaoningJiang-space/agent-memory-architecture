"""Record both GPUs' gross energy/power; never infer CPU or wall-plug energy."""
import json
import signal
import sys
import time
import pynvml

pynvml.nvmlInit()
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
with open(sys.argv[1], 'a', buffering=1) as f:
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    while running:
        for i,h in enumerate(handles):
            row = {'time':time.time(), 'gpu':i, 'power_w':pynvml.nvmlDeviceGetPowerUsage(h)/1000,
                   'memory_used':pynvml.nvmlDeviceGetMemoryInfo(h).used,
                   'utilization':pynvml.nvmlDeviceGetUtilizationRates(h).gpu}
            try:
                row['energy_j'] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)/1000
            except pynvml.NVMLError:
                row['energy_j'] = None
            f.write(json.dumps(row)+'\n')
        time.sleep(0.1)
