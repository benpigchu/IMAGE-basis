import numpy
import sys
import cv2

def mapped(func):
	def process(image,value):
		height,width,=(image.shape)
		for i in range(height):
			for j in range(width):
				image[i,j]=max([0,min([func(image[i,j],value),255])])
		return image
	return process

def equalize(image):
	height,width,=(image.shape)
	histogram=[0 for i in range(256)]
	mapper=[0 for i in range(256)]
	for i in range(height):
		for j in range(width):
			histogram[image[i,j]]+=1
	current=0
	smallest=0
	for i in range(256):
		current+=histogram[i]
		if (smallest==0)and(current!=0):
			smallest=current
		mapper[i]=(255*(current-smallest))//(height*width-smallest)
	for i in range(height):
		for j in range(width):
			image[i,j]=max([0,min([mapper[image[i,j]],255])])
	return image

def match(image,target):
	height,width,=(image.shape)
	theight,twidth,=(target.shape)
	histogram=[0 for i in range(256)]
	targethisto=[0 for i in range(256)]
	mapper=[0 for i in range(256)]
	for i in range(height):
		for j in range(width):
			histogram[image[i,j]]+=1
	for i in range(theight):
		for j in range(twidth):
			targethisto[target[i,j]]+=1
	for i in range(255):
		histogram[i+1]+=histogram[i]
		targethisto[i+1]+=targethisto[i]
	for i in range(256):
		histogram[i]/=height*width
		targethisto[i]/=theight*twidth
	last=0
	for i in range(256):
		while True:
			if targethisto[last]>=histogram[i]:
				mapper[i]=last
				break
			last+=1
	for i in range(height):
		for j in range(width):
			image[i,j]=max([0,min([mapper[image[i,j]],255])])
	return image

def IVIwarpper(func):
	def warpped(image,value,output):
		cv2.imwrite(output,func(cv2.imread(image,cv2.IMREAD_GRAYSCALE),float(value)))
		return
	return warpped

def IIwarpper(func):
	def warpped(image,output):
		cv2.imwrite(output,func(cv2.imread(image,cv2.IMREAD_GRAYSCALE)))
		return
	return warpped

def IIIwarpper(func):
	def warpped(image1,image2,output):
		cv2.imwrite(output,func(cv2.imread(image1,cv2.IMREAD_GRAYSCALE),cv2.imread(image2,cv2.IMREAD_GRAYSCALE)))
		return
	return warpped

operator={
	"brightness":IVIwarpper(mapped(lambda x,v:x+v)),
	"contrast":IVIwarpper(mapped(lambda x,v:127+(x-127)*v)),
	"gamma":IVIwarpper(mapped(lambda x,v:255*(x/255)**v)),
	"equalize":IIwarpper(equalize),
	"match":IIIwarpper(match)
}

_,operate,*params=sys.argv
operator[operate](*params)
