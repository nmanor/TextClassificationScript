import os
import sys

if __name__ == '__main__':
	output_path = sys.argv[1]
	for folder in ['cfgs', 'results', 'dataset', 'output']:
		os.mkdir(os.path.join(output_path, folder))
		print('Creating '+folder+' folder...')

	for folder in ['training', 'testing']:
		os.mkdir(os.path.join(output_path, 'dataset', folder))
		print('Creating '+folder+' folder...')
