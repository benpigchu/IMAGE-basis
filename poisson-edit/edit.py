import numpy
import scipy.sparse
import scipy.sparse.linalg
import sys
import cv2
import math

def edit(src,mask,dst,offsetx,offsety):
	src_region=(max([-offsety,0]),max([-offsetx,0]),min([dst.shape[0]-offsety,src.shape[0]]),min([dst.shape[1]-offsetx,src.shape[1]]))
	dst_region=(src_region[0]+offsety,src_region[1]+offsetx,src_region[2]+offsety,src_region[3]+offsetx)
	region_size=(src_region[2]-src_region[0],src_region[3]-src_region[1])
	mask=mask[src_region[0]:src_region[2],src_region[1]:src_region[3]]
	A=scipy.sparse.identity(region_size[0]*region_size[1],format='lil')
	for y in range(region_size[0]):
		for x in range(region_size[1]):
			if mask[y,x]>50:
				index=x+y*region_size[1]
				A[index,index]=4
				if index+1<region_size[0]*region_size[1]:
					A[index,index + 1]=-1
				if index-1>=0:
					A[index,index-1]=-1
				if index+region_size[1]<region_size[0]*region_size[1]:
					A[index,index+region_size[1]]=-1
				if index-region_size[1]>=0:
					A[index,index-region_size[1]]=-1
	A=A.tocsr()
	for channel in range(dst.shape[2]):
		s=src[src_region[0]:src_region[2],src_region[1]:src_region[3],channel].flatten()
		d=dst[dst_region[0]:dst_region[2],dst_region[1]:dst_region[3],channel].flatten()
		B=A@s
		for y in range(region_size[0]):
			for x in range(region_size[1]):
				if not(mask[y,x]>50):
					index=x+y*region_size[1]
					B[index]=d[index]
		result,*_=scipy.sparse.linalg.lsqr(A,B)
		result=numpy.reshape(result,region_size)
		result[result>255]=255
		result[result<0]=0
		dst[dst_region[0]:dst_region[2],dst_region[1]:dst_region[3],channel]=result
	return dst

def IIIVVIwarpper(func):
	def warpped(image1,image2,image3,value1,value2,output):
		cv2.imwrite(output,func(cv2.imread(image1),cv2.imread(image2,cv2.IMREAD_GRAYSCALE),cv2.imread(image3),int(value1),int(value2)))
		return
	return warpped

_,*params=sys.argv
IIIVVIwarpper(edit)(*params)