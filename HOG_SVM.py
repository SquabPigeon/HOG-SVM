'''
1.通过txt文件加载正样本与负样本的路径，并更改为绝对路径
2.对正负样本依次加载，初始化hog算子，计算hog特征并设置标志label
3.初始化SVM，并将hog特征与label带入进行train，训练完毕后保存模型
4.加载模型，使用test数据测试
'''

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
#from nms import py_cpu_nms
#from imutils.object_detection import non_max_suppression
def is_inside(o, i):
	'''
	判断矩形o是不是在i矩形中

	args:
		o：矩形o  (x,y,w,h)
		i：矩形i  (x,y,w,h)
	'''
	ox, oy, ow, oh = o
	ix, iy, iw, ih = i
	return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def writeFilename2txt(filename):
	'''
	:return:
	'''
	rootdir = filename
	list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
	jpgfile = []
	for i in range(0, len(list)):
		single_filename = list[i]
		if single_filename[-3:] == 'bmp':
			jpgfile.append(single_filename)
	filename_write = filename + 'total.txt'
	with open(filename_write,'w') as f:
		print('开始写入')
		for file in jpgfile:
			f.write(file)
			f.write('\n')
		print('写入完毕')

def readfile(filename):
	'''

	:return:
	'''
	filenametxt = filename + '\\total.txt'
	img_list = []
	with open(filenametxt,'r') as f:
		listLines = f.readlines();  # 将所有信息读到list中
		for i in range(0, len(listLines)):
			listLines[i] = listLines[i].rstrip('\n')  # 去掉换行符
			img_tmp_filename = 	filename + '\\' + listLines[i]
			img_temp = cv2.imread(img_tmp_filename)
			img_list.append(img_temp)
	return img_list





def LoadSample(img_list,label):
	'''

	:param img_list:
	:return: hog_vector_list,label_list返回所有样本的hog特征以及label
	'''

	'''
	hog初始化
	'''
	winSize = (64,128)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9

	hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
	hog_vector_list = []
	labels = []
	label = label + '.jpg'
	for img in img_list:
		hist_tmp = hog.compute(img, winStride = (64,128), padding=(0, 0))
		hog_vector_list.append(hist_tmp)
		labels.append(label)

	return hog_vector_list, labels

def SVMInitial():
	'''

	:return: svm,返回svm算子
	'''
	svm = cv2.ml.SVM_create()
	svm.setCoef0(0)
	svm.setCoef0(0.0)
	svm.setDegree(3)
	criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
	svm.setTermCriteria(criteria)
	svm.setGamma(0)
	svm.setKernel(cv2.ml.SVM_LINEAR)
	svm.setNu(0.5)
	svm.setP(0.1)
	svm.setC(0.01)
	svm.setType(cv2.ml.SVM_EPS_SVR)

	return svm



def get_svm_detector(svm):
	'''
	获取svm参数
	:param svm:
	:return:
	'''
	sv = svm.getSupportVectors()
	rho, _, _ = svm.getDecisionFunction(0)
	sv = np.transpose(sv)
	return np.append(sv,[[-rho]],0)


#def train(svm,positive_simple,positive_label,negative_simple,negative_label):
def train():
	'''

	:param svm:
	:param positive_simple:
	:param positive_label:
	:param negative_simple:
	:param negative_label:
	:return: bin,返回并保存模型
	'''
	'''
	初始化hog描述子
	'''
	winSize = (64,128)   #特征图大小
	blockSize = (16,16)  #块（block）大小，2*2个cell
	blockStride = (8,8)  #步长，一个cell大小
	cellSize = (8,8)     #cell大小
	nbins = 9            #一个cell中bin特征向量个数

	hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
	'''
	读取正样本 ，并生成HOG特征
	'''
	filename_positive = 'H:\\SVM_HOG\\positive'
	filenametxt_positive = 'H:\\SVM_HOG\\imgfilename\\pos-total.txt'
	positive_simple = []
	positive_label = []
	with open(filenametxt_positive,'r') as f:
		listLines = f.readlines();  # 将所有信息读到list中
		for i in range(0, len(listLines)):
			listLines[i] = listLines[i].rstrip('\n')  # 去掉换行符
			img_temp_filename = 	filename_positive + '\\' + listLines[i]#获取绝对路径
			img_temp = cv2.imread(img_temp_filename)
			assert img_temp is not None
			img_temp_resize = cv2.resize(img_temp,dsize=(64,128),interpolation = cv2.INTER_AREA)#修剪成64*128
			descriptor = []
			descriptor = hog.compute(img_temp_resize)
			print('正样本',listLines[i],'处理完毕\n')
			#print(descriptor.shape) #(3780,1)
			positive_simple.append(descriptor)
			positive_label.append(1)#1表示有

	'''
	读取负样本 ，并生成HOG特征
	'''
	filename_negative = 'H:\\SVM_HOG\\negative'
	filenametxt_negative = 'H:\\SVM_HOG\\imgfilename\\neg-total.txt'
	negative_simple = []
	negative_label = []
	with open(filenametxt_negative,'r') as f:
		listLines2 = f.readlines();  # 将所有信息读到list中
		for i in range(0, len(listLines2)):
			listLines2[i] = listLines2[i].rstrip('\n')  # 去掉换行符
			img_temp_filename = 	filename_negative + '\\' + listLines2[i]#获取绝对路径
			img_temp = cv2.imread(img_temp_filename)
			assert img_temp is not None
			#print('开始处理：',listLines2[i],'\n')
			img_temp_resize = cv2.resize(img_temp,dsize=(64,128),interpolation = cv2.INTER_AREA)#修剪成64*128
			descriptor = []
			descriptor = hog.compute(img_temp_resize)
			print('负样本',listLines2[i],'处理完毕\n')
			#print(descriptor.shape) #(3780,1)
			negative_simple.append(descriptor)
			negative_label.append(-1)#1表示有
		simples = positive_simple + negative_simple
		labels = positive_label +negative_label
		'''至此所有正负样本特征获取完毕'''
	'''初始化svm算子'''
	svm = SVMInitial()
	print('svm training...')
	svm.train(np.array(simples),cv2.ml.ROW_SAMPLE,np.array(labels))
	print('svm training complete...')
	svm.save('SVM_HOG.xml')
	'''获取当前训练出的分类器'''
	Descriptor = get_svm_detector(svm)
	#print(Descriptor)
	with open('svm_dector.txt','w') as f:
		for i in Descriptor:
			f.write(str(i[0]))
			f.write('\n')
	'''替换hog中自带的分类器'''
	hog.setSVMDetector(Descriptor)
	hog.save('myHogDector.bin')
	print('分类器替换完毕')


	'''
	测试
	'''
	img = cv2.imread('img/test (1).bmp',0)
	assert img is not None


	rects, scores = hog.detectMultiScale(img, winStride=(2,16), padding=(20,20), scale=1.5)
	'''
	HOG detectMultiScale 参数分析 见：https://www.cnblogs.com/klitech/p/5747895.html
	winStride:HoG检测窗口移动时的步长(水平及竖直),
	          winStride和scale都是比较重要的参数，需要合理的设置。一个合适参数能够大大提升检测精确度，同时也不会使检测时间太长
	padding:在原图外围添加像素，作者在原文中提到，适当的pad可以提高检测的准确率（可能pad后能检测到边角的目标？）
	         常见的pad size 有(8, 8), (16, 16), (24, 24), (32, 32).
	scale :如图是一个图像金字塔，也就是图像的多尺度表示。每层图像都被缩小尺寸并用gaussian平滑。
	       scale参数可以具体控制金字塔的层数，参数越小，层数越多，检测时间也长。 一下分别是1.01  1.5 1.03 时检测到的目标。 通常scale在1.01-1.5这个区间
	    
	
	'''
	sc = [score[0] for score in scores]
	sc = np.array(sc)

	# 转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
	for i in range(len(rects)):
		r = rects[i]
		rects[i][2] = r[0] + r[2]
		rects[i][3] = r[1] + r[3]

	pick = []
	'''
	# 非极大值移植
	print('rects_len', len(rects))
	pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
	print('pick_len = ', len(pick))
	'''

	# 画出矩形框
	# 过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
	found_filtered = []
	for ri, r in enumerate(rects):
		for qi, q in enumerate(rects):
			# r在q内？
			if ri != qi and is_inside(r, q):
				break
		else:
			found_filtered.append(r)
	for (x, y, xx, yy) in found_filtered:
		cv2.rectangle(img, (x, y), (xx, yy), (0, 0, 255), 2)

	cv2.imshow('a', img)
	cv2.waitKey(0)

	'''
	hog.setSVMDetector(get_svm_detector(svm))
	hog.save('myHogDector.bin')
	'''

	'''
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;

    将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。

    如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），

    就可以利用你的训练样本训练出来的分类器进行行人检测了。
	'''
	#DescriptorDim = svm.get_var_count(); #特征向量的维数，即HOG描述子的维数


def test():
	'''
	加载模型并test
	:param bin:
	:return: 返回测试结果
	'''

	hog = cv2.HOGDescriptor()
	#hog.load('myHogDector.bin')
	Descriptor = []
	with open('svm_dector.txt','r+') as f:
		lines = f.readlines();
		for line in lines:
			line.rstrip('\n')
			D_temp = [float(line)]
			Descriptor.append(D_temp)

	hog.setSVMDetector(np.array(Descriptor))


	src = cv2.imread('img/test (63).bmp')
	assert src is not None

	img = src[0:int(src.shape[0]/4),0:int(src.shape[1])]

	rects, scores = hog.detectMultiScale(img, winStride=(8, 8), padding=(0, 0), scale=1.05)

	sc = [score[0] for score in scores]
	sc = np.array(sc)

	# 转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
	for i in range(len(rects)):
		r = rects[i]
		rects[i][2] = r[0] + r[2]
		rects[i][3] = r[1] + r[3]

	pick = []
	'''
	# 非极大值移植
	print('rects_len', len(rects))
	pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
	print('pick_len = ', len(pick))
	'''

	# 画出矩形框
	# 过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
	found_filtered = []
	for ri, r in enumerate(rects):
		for qi, q in enumerate(rects):
			# r在q内？
			if ri != qi and is_inside(r, q):
				break
		else:
			found_filtered.append(r)
	for (x, y, xx, yy) in found_filtered:
		cv2.rectangle(img, (x, y), (xx, yy), (0, 255, 0), 3)
		break

	cv2.imshow('a', img)
	cv2.waitKey(0)

if __name__ == '__main__':
	#writeFilename2txt('H:\\SVM_HOG\\test\\')
	#readfile('H:\\SVM_HOG\\positive')
	#train()
	test()
