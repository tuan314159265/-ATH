import matplotlib.pyplot as plt

quantities = [12,24,48,24,18,24,24,24,24,20,24,24,12,24,24,36,3,80,6,6,6,6,8,6,6,6,6,6,2,4]

plt.hist(quantities, bins=6, edgecolor='black')
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.title("Histogram of Quantity (6 bins)")
plt.show()