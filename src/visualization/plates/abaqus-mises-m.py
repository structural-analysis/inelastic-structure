from abaqus import *
from abaqusConstants import *
import numpy as np
import visualization

# Get the current ODB (Output Database) object
odb = session.odbs['C:/temp/fine-flats-free.odb']

# Access the last frame of the last step
lastFrame = odb.steps['Step-1'].frames[92]

SM_field = lastFrame.fieldOutputs['SM']
von_mises_values = []

for value in SM_field.values:
    mx = value.data[1]  # SM1 (Mx)
    my = value.data[0]  # SM2 (My)
    mxy = value.data[2]  # SM3 (Mxy)
    von_mises = np.sqrt(mx**2 + my**2 - mx * my + 3 * mxy**2)
    print(von_mises)
    von_mises_values.append(von_mises)

vonMisesFieldOutput = lastFrame.FieldOutput(
    name='VON_MISES_MOMENTS_NEW',
    description='Von Mises moments of plate element',
    type=SCALAR
)

# Add calculated values to the new field output
for i, value in enumerate(SM_field.values):
    vonMisesFieldOutput.addData(
        position=value.position,
        instance=value.instance,
        elementLabel=value.elementLabel,
        integrationPoint=value.integrationPoint,
        data=von_mises_values[i]
    )

# Save the updated ODB
odb.save()