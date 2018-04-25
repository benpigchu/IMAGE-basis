import numpy
import sys
import cv2
import math
import detector

def morph(image1,image2):
	height1,width1,_=(image1.shape)
	height2,width2,_=(image2.shape)
	points1=detector.getPoints(image1)
	points2=detector.getPoints(image2)
	points1.append((0,0))
	points2.append((0,0))
	points1.append((width1-1,height1-1))
	points2.append((width2-1,height2-1))
	points1.append((0,height1-1))
	points2.append((0,height2-1))
	points1.append((width1-1,0))
	points2.append((width2-1,0))
	points1.append((width1//2,height1-1))
	points2.append((width2//2,height2-1))
	points1.append((width1//2,0))
	points2.append((width2//2,0))
	points1.append((width1-1,height1//2))
	points2.append((width2-1,height2//2))
	points1.append((0,height1//2))
	points2.append((0,height2//2))
	print(height1,width1,points1)
	print(height2,width2,points2)
	heightt=(height1+height2)//2
	widtht=(width1+width2)//2
	pointst=[((p1[0]+p2[0])//2,(p1[1]+p2[1])//2) for p1,p2 in zip(points1,points2)]
	pointst=[(max([0,min([y,widtht-1])]),max([0,min([x,heightt-1])])) for y,x in pointst]
	print(heightt,widtht,pointst)
	rect=(0,0,widtht,heightt)
	print(rect)
	subdiv=cv2.Subdiv2D(rect)# this is geometry calculation, not image processing
	ptid={}
	for i,pt in enumerate(pointst):
		ptid[subdiv.insert(pt)]=i
	triangleList=subdiv.getTriangleList()
	print(triangleList)
	output=numpy.zeros((heightt,widtht,3),dtype=numpy.uint8)
	print(output.shape)
	print(image1.shape)
	print(image2.shape)
	output[heightt-1,widtht-1]=(255,255,255)
	def getPt(x,y):
		if not((0<=x<widtht)and(0<=y<heightt)):
			return -1
		id,_=subdiv.findNearest((x,y))
		return ptid[id]
	def setOutput(x,y,color):
		if not((0<=x<widtht)and(0<=y<heightt)):
			return
		output[y,x]=(color)
	def getColor(image,pos):
		y,x=pos
		y,x=int(y),int(x)
		y,x=(max([0,min([y,image.shape[1]-1])]),max([0,min([x,image.shape[0]-1])]))
		return image[x,y]
	def paint(x,y,p1,p2,p3):
		p,q,r=numpy.linalg.solve(((pointst[p1][0],pointst[p2][0],pointst[p3][0]),(pointst[p1][1],pointst[p2][1],pointst[p3][1]),(1,1,1)),(x,y,1))
		color=(p*255,q*255,r*255)
		if not((-0.001<=p<=1.001)and(-0.001<=q<=1.001)and(-0.001<=r<=1.001)):
			return
		pos1=numpy.matmul(((points1[p1][0],points1[p2][0],points1[p3][0]),(points1[p1][1],points1[p2][1],points1[p3][1])),(p,q,r))
		pos2=numpy.matmul(((points2[p1][0],points2[p2][0],points2[p3][0]),(points2[p1][1],points2[p2][1],points2[p3][1])),(p,q,r))
		color1=getColor(image1,pos1)
		color2=getColor(image2,pos2)
		setOutput(x,y,color1/2+color2/2)
	def draw(p1,p2,p3):
		p1,p2,p3=sorted((p1,p2,p3),key=lambda p:pointst[p][1])
		if (pointst[p2][1]-pointst[p1][1])==0:
			dx12=pointst[p2][0]-pointst[p1][0]
		else:
			dx12=(pointst[p2][0]-pointst[p1][0])/(pointst[p2][1]-pointst[p1][1])
		if (pointst[p3][1]-pointst[p1][1])==0:
			dx13=pointst[p3][0]-pointst[p1][0]
		else:
			dx13=(pointst[p3][0]-pointst[p1][0])/(pointst[p3][1]-pointst[p1][1])
		if (pointst[p3][1]-pointst[p2][1])==0:
			dx23=pointst[p3][0]-pointst[p2][0]
		else:
			dx23=(pointst[p3][0]-pointst[p2][0])/(pointst[p3][1]-pointst[p2][1])
		y=pointst[p1][1]
		l=pointst[p1][0]
		r=pointst[p1][0]
		def hzline(l,r,y):
			x=math.floor(l)-2
			while x<=math.ceil(r)+2:
				paint(x,y,p1,p2,p3)
				x+=1
		if dx12>dx13:
			while y<=pointst[p2][1]:
				hzline(l,r,y)
				l+=dx13
				r+=dx12
				y+=1
			y=pointst[p2][1]
			r=pointst[p2][0]
			while y<=pointst[p3][1]:
				hzline(l,r,y)
				l+=dx13
				r+=dx23
				y+=1
		else:
			while y<=pointst[p2][1]:
				hzline(l,r,y)
				r+=dx13
				l+=dx12
				y+=1
			y=pointst[p2][1]
			l=pointst[p2][0]
			while y<=pointst[p3][1]:
				hzline(l,r,y)
				r+=dx13
				l+=dx23
				y+=1
	for x1,y1,x2,y2,x3,y3 in triangleList:
		p1=getPt(x1,y1)
		p2=getPt(x2,y2)
		p3=getPt(x3,y3)
		if (p1<0)or(p2<0)or(p3<0):
			continue
		draw(p1,p2,p3)
	return output

def IIIwarpper(func):
	def warpped(image1,image2,output):
		cv2.imwrite(output,func(cv2.imread(image1),cv2.imread(image2)))
		return
	return warpped

_,*params=sys.argv
IIIwarpper(morph)(*params)