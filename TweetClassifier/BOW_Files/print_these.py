import os
for filename in os.listdir('.'):
	if filename.endswith('.txt'):
		contents = []
		with open(filename) as f:
			for line in f:
				contents.append(line.strip())
		print(filename, ":", contents)
		print()