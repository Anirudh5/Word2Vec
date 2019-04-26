import numpy as np
f = open("naturedata.txt")
lines = []
for line in f:
	line = line.strip()
	lines.append(line)
l = [i for i in range(len(lines))]
l = np.array_split(l,15)
for i in range(1,16):
	data = ""
	for id in l[i-1]:
		data = data + lines[id] + "\n"
	f1 = open("naturedata"+str(i)+".txt",'w')
	f1.write(data)
	f1.close()
	print(i)
