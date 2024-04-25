from matplotlib import pyplot as plt

x3 = [1,2,3,4,5,6,7,8]
y3 = [1,23,4,5,6,7,8,4]
y4 = [4,6,7,8,9,2,1,4]

# create two lists of (x, y) pairs using zip()
line1 = list(zip(x3, y3))
line2 = list(zip(x3, y4))

# plot the lines separately with their own settings and label
plt.plot([p[0] for p in line1], [p[1] for p in line1], color='green', marker='.', linestyle='solid', linewidth=1, markersize=5, label='Line 1')
plt.plot([p[0] for p in line2], [p[1] for p in line2], color='red', marker='.', linestyle='solid', linewidth=1, markersize=5, label='Line 2')

# add a legend to the plot
plt.legend()

plt.show()