import os
import numpy as np
import matplotlib.pyplot as plt

from src.settings import settings

element = "0"
dof = "5"

outputs_dir = "output/examples/"
example_name = settings.example_name
example_path = os.path.join(outputs_dir, example_name)
aggregate_path = os.path.join(example_path, "aggregatation")
response_path = os.path.join(aggregate_path, "members_nodal_forces")
element_path = os.path.join(response_path, element)
dof_path = os.path.join(element_path, f"{dof}.csv")

time = np.zeros((51))
for i in range(51):
    time[i] = i * 0.02
resp = np.loadtxt(fname=dof_path, usecols=range(1), delimiter=",", ndmin=1, skiprows=0, dtype=float)

plt.plot(time, resp)

plt.show()
