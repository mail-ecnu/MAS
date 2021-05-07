import matplotlib.pyplot as plt

with open('rewards.log', 'r') as f:
    acc = [float(line.split(':')[-1]) for line in f.readlines()]

plt.plot(acc)
plt.savefig('acc.png')
