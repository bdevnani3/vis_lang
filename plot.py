import os, sys
import matplotlib.pyplot as plt
import json
from utils import init_root

def load_data_file(path:str):
	with open(path, 'r', encoding='ascii') as f:
		text = f.read()
	
	try:
		return json.loads(text)
	except:
		ret = {}
		for line in text.split('\n'):
			line = line.strip()
			if len(line) > 1:
				k, v = line.split(': ')
				ret[k] = [float(x) for x in v[1:-1].split(', ')]
		return ret


def plot(axis, models):
	root_dir = init_root()

	plt.title(axis)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy (%)' if 'Acc' in axis else 'Loss')

	for model in models:
		data = load_data_file(os.path.join(root_dir, 'vis_lang', model, 'raw_data'))[axis]
		plt.plot(list(range(len(data))), data, label=model)

	plt.legend()
	plt.show()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: python plot.py <name_of_axis> <model1> <model2> ...')
		exit()

	axis = sys.argv[1]
	models = sys.argv[2:]

	plot(axis, models)
