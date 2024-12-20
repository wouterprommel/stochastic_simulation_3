import pandas as pd
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

df = pd.read_csv('results.csv')

df1= df[df['N'] == 22]
print(df1)

m_list = []
for i in range(12, 41):
    dfn = df[df['N'] == i]
    id = dfn['E'].idxmin()
    m = dfn.loc[id]['Middle']
    m_list.append(m)

ref_N = [11, 12, 16, 17, 18, 19, 20, 21, 22, 23]
ref_M = [0, 1, 2, 3, 3, 3, 3, 4, 5, 5]

plt.scatter(ref_N, ref_M, marker='v', s=100, color='tab:green', label='Reference')
plt.scatter(list(range(12, 41)), m_list,  marker='^', s=100, color='tab:blue', label='Simulation')

plt.xticks(list(range(11, 42, 2)))
plt.yticks(list(range(0, 14)))
plt.grid(True)
plt.legend()
plt.xlabel('Total number of charges')
plt.ylabel('Number of charges in middle')
plt.savefig('driehoeks.pdf', format='pdf', bbox_inches='tight')
plt.show()