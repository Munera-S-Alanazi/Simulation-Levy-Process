import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# The result from VG simulation, First read data from CSV file
df = pd.read_csv('vg_results.csv')

# here I extract data from the dataframe
zValues_VG = df['zValues']
simulated_real_VG = df['Simulated Real']
theoretical_real_VG = df['Theoretical Real']
absolute_error_VG = df['Absolute Error']
three_std_dev_VG = df['3 Std Dev']



# Plot real component of CF with error bars for MC simulation
plt.figure()
plt.plot(zValues_VG, simulated_real_VG, label='Simulated Real Part')
plt.errorbar(zValues_VG, simulated_real_VG, yerr=three_std_dev_VG, fmt='o', label='Simulated Real Part (with error)', ecolor='red', capsize=5, elinewidth=2)
plt.plot(zValues_VG, theoretical_real_VG, label='Theoretical Real Part', linestyle='--')
plt.title('Real Component of Characteristic Function (VG Process)')
plt.xlabel('z')
plt.ylabel('Real Part')
plt.legend()
plt.grid(True)

# Plot absolute error and 3 standard deviations
plt.figure()
plt.plot(zValues_VG, absolute_error_VG, label='Absolute Error')
plt.plot(zValues_VG, three_std_dev_VG, label='3 Standard Deviations')
plt.title('Error in Real Component of VG Characteristic Function')
plt.legend()
plt.grid(True)


# The result from NIG simulation, read data from CSV file
df_NIG = pd.read_csv('NIG_results.csv')

# here I extract data from the dataframe
zValues_NIG = df_NIG['zValues']
simulated_real_NIG = df_NIG['Simulated Real']
theoretical_real_NIG = df_NIG['Theoretical Real']
absolute_error_NIG = df_NIG['Absolute Error']
three_std_dev_NIG = df_NIG['3 Std Dev']



# Plot real component of CF with error bars for MC simulation
plt.figure()
plt.plot(zValues_NIG, simulated_real_NIG, label='Simulated Real Part')
plt.errorbar(zValues_NIG, simulated_real_NIG, yerr=three_std_dev_NIG, fmt='o', label='Simulated Real Part (with error)', ecolor='red', capsize=5, elinewidth=2)
plt.plot(zValues_NIG, theoretical_real_NIG, label='Theoretical Real Part', linestyle='--')
plt.title('Real Component of Characteristic Function (NIG Process)')
plt.xlabel('z')
plt.ylabel('Real Part')
plt.legend()
plt.grid(True)

# Plot absolute error and 3 standard deviations
plt.figure()
plt.plot(zValues_NIG, absolute_error_NIG, label='Absolute Error')
plt.plot(zValues_NIG, three_std_dev_NIG, label='3 Standard Deviations')
plt.title('Error in Real Component of NIG Characteristic Function')
plt.legend()
plt.grid(True)

plt.show()
