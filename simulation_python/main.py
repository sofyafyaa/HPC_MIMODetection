import yaml
import numpy as np
from simulation import MIMOOFDMSimulation

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
config['snr_db_range'] = list(np.arange(-10, 40, 5))

sim = MIMOOFDMSimulation(config)
sim.simulate()
