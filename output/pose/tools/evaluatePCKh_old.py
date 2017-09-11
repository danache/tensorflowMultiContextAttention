import numpy as np
import csv

def readData(path, gt_label):
	labels = []
	with open(path,'rb') as csvfile:
		reader = csv.reader(csvfile,delimiter=',')
		for row in reader:
			label = row[1:]
                        # print(row[0]) 
			for l in range(len(label)):
				if label[l] == 'nan':
					label[l] = '-1'
				label[l] = float(label[l])
			labels.append(label)
	data = np.array(labels)
	dim =2
	if gt_label:
		dim = 3
	data = np.reshape(data,[data.shape[0], data.shape[1]/dim, dim])
	if gt_label:
		data = data[:,:,0:2]
	else:
		data[data<0] = 1
	return data

def getHeadSize(gt):
	headSize = np.linalg.norm(gt[:,9,:] - gt[:,8,:],axis=1)
	for n in range(gt.shape[0]):
		if gt[n,8,0] < 0 or gt[n,9,0] < 0:  #invalid gt head size 
			headSize[n] = 0
			
	return headSize

def getDistPCKh(pred,gt,headSize):
	# pred shape NxPx2
	# gt shape   NxPx2
	# headSize shape Nx1
	N = pred.shape[0]
	P = pred.shape[1]
	dist = np.zeros([N,P])
	for n in range(N):
		refDist = headSize[n]
		if refDist == 0:
			dist[n,:] = -1   # invalid gt head size
		else:
			
			dist[n,:] = np.linalg.norm(gt[n,:,:] - pred[n,:,:],axis = 1) / refDist
			for p in range(P):
				if gt[n,p,0] < 0 or gt[n,p,1] < 0:   # invalid gt points
					dist[n,p] = -1
	return dist

def computePCK(dist,threshRange):
	P = dist.shape[1]
	pck = np.zeros([len(threshRange),P+2])

	for p in range(P):
		for k_ind in range(len(threshRange)):
			k = threshRange[k_ind]
			joint_dist = dist[:,p]
			# print(joint_dist[np.where(joint_dist>0)])
			pck[k_ind,p] = 100 * np.mean(joint_dist[np.where(joint_dist>=0)] <= k)
	# uppper body
	for k_ind in range(len(threshRange)):
		k = threshRange[k_ind]
		joint_dist = dist[:,8:16]
		pck[k_ind,P] = 100 * np.mean(joint_dist[np.where(joint_dist>=0)] <= k)
	
	# total joints
	for k_ind in range(len(threshRange)):
		k = threshRange[k_ind]
		joints_index = range(0,6) + range(8,16)

		joint_dist = dist[:,joints_index]
		
		pck[k_ind,P+1] = 100 * np.mean(joint_dist[np.where(joint_dist>=0)] <= k)
	return pck

def genTablePCK(pck,name):
	print('PCKh@0.5 & Head & Shoulder & Elbow & Wrist & Hip & Knee  & Ankle & UBody & Total \\ \n')
	print('%s& %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f  & %1.1f & %1.1f & %1.1f & %1.1f %s\n'%(name, 
						(pck[8]+pck[9])/2,(pck[12]+pck[13])/2,(pck[11]+pck[14])/2,(pck[10]+pck[15])/2,(pck[2]+pck[3])/2,
							(pck[1]+pck[4])/2,(pck[0]+pck[5])/2,pck[-2],pck[-1],'\\'))


def evaluatePCKh(pred_path,gt_path,order_to_lsp,name):
	pred = readData(pred_path, False)
	pred = pred[:,order_to_lsp]
	gt = readData(gt_path,True)
	assert gt.shape[0] == pred.shape[0], 'sample not matched'
	assert gt.shape[1] == pred.shape[1], 'joints not matched'
	assert gt.shape[2] == pred.shape[2], 'dim not matched'
	# print('Total test sample', gt.shape[0])
	# print('Total joints',gt.shape[1])

	threshRange = np.arange(0.50, 0.51, 0.01)
	headSize = getHeadSize(gt)

	dist = getDistPCKh(pred,gt,headSize)
	
	pck = computePCK(dist,threshRange)
	genTablePCK(pck[-1], name)
	return pck

if __name__ == "__main__":
	gt_path = './lip_test_set_2000.csv'
	order_to_lsp = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	pred_path = './csv/dengjia_predict.csv'
	evaluatePCKh(pred_path,gt_path, order_to_lsp,'dengjia')
	pred_path = './csv/deep_predict.csv'
	evaluatePCKh(pred_path,gt_path, order_to_lsp,'deepcut')
	pred_path = '../test_result.csv'
	evaluatePCKh(pred_path, gt_path, order_to_lsp, 'gan')
