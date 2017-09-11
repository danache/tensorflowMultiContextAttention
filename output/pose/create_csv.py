import csv
import os

root_folder = 'test'
save_csv_path = 'test_result.csv'
anno_list_str = []
for root, dirs, files in os.walk(root_folder):
	for name in files:
		anno_list = name[:-4] + '.jpg'
		fi = open(os.path.join(root, name), 'r')
		lines = fi.readlines()
		for point in lines[0].split(' '):
			anno_list += ',' + point
		anno_list_str.append(anno_list[:-1])

with open(save_csv_path, 'w') as save_csv_file:
	save_csv_file.write('\n'.join(sorted(anno_list_str)))


