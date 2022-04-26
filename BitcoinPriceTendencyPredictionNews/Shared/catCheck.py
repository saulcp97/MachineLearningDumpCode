import matplotlib.pyplot as plt
from utils import *
import numpy as np

categories = extractV('category')
categoryK = list(set(categories))
values = [categories.count(i) for i in categoryK]

dictionary = {category:categories.count(category) for category in categoryK}

#print(dictionary)

fig, ax = plt.subplots()
cat = categoryK
occ = values

for label in ax.get_yticklabels():
	label.set_fontsize(7.5)

ax.barh(cat, occ)
plt.rc('axes', labelsize=59)
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y(),
        str(round((i.get_width()), 2)),
        fontsize = 7, fontweight ='bold',
        color ='grey')
plt.show()