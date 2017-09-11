import csv
import sys
import random
import os
from PIL import Image, ImageDraw

def plot_joint(rec, img_folder):
	img_name = os.path.join(img_folder, 'images',rec[0])
	print('Image at: ' + img_name)
	
	img = Image.open(img_name)
	draw = ImageDraw.Draw(img)
	r = 5
	bombs = [[0,1],[1,2]
			,[3,4],[4,5]
			,[6,7],[7,8],[8,9]
			,[10,11],[11,12]
			,[13,14],[14,15] ]
	colors = [(255,0,0),(255,0,0),
			  (0,255,0),(0,255,0),
			  (0,0,255),(0,0,255),(0,0,255),
			  (128,128,0),(128,128,0),
			  (128,0,128),(128,0,128)]
	r = 5 
	for b_id in range(len(bombs)):
		b = bombs[b_id]
		color = colors[b_id]
		x1 = rec[ b[0] * 2 + 1]
		y1 = rec[ b[0] * 2 + 2]
		
		x2 = rec[ b[1] * 2 + 1]
		y2 = rec[ b[1] * 2 + 2]
		
		if x1 > 0 and x2 > 0 :
			draw.line((int(x1),int(y1), int(x2),int(y2)), fill = color, width = 5)
		elif x1 > 0:
			draw.ellipse((int(x1) - r, int(y1) - r, int(x1) + r, int(y1) + r), fill = color)
		elif x2 > 0:
			draw.ellipse((int(x2) - r, int(y2) - r, int(x2) + r, int(y2) + r), fill = color)

	img.show()

def vis_anno(dataSet):
	csv_path = {
		
		'test' :'lip_predict.csv',
	}
	img_root = {
		
		'test': '../LIP_dataset/test_set',
	}
	with open(csv_path[dataSet], 'rb') as f:
		reader = csv.reader(f)
		recs = []
		for row in reader:
			recs.append(row)
		random_id = random.randint(0, len(recs) - 1)
		plot_joint(recs[random_id], img_root[dataSet])

def error():
	print('Error!')
	print('Usage: python vis_annotation.py [test]')
	sys.exit()

if __name__ == "__main__":
	
	dataSet = 'test'

	if len(sys.argv) == 2:
		dataSet = sys.argv[1].lower()
		print dataSet
		if dataSet not in ['train', 'valid', 'test']:
			error()
			
	elif len(sys.argv) > 2:
		error()

	vis_anno(dataSet)
