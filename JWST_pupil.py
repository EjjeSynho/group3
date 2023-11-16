# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:19:40 2023
​
@author: stars
"""
​
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:42:44 2023
​
@author: stars
"""
​
import matplotlib.pyplot as mp
# from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from astropy.io import fits
import math as math
import cmath as cmath
import numpy as np
import imageio
import PIL
from PIL import Image
import pickle
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import astropy.units as u
from scipy.optimize import curve_fit
​
N=256# total size of array
​
n_ring=2#number of rings
n_seg=n_ring*(3*(n_ring+1))+1#total number of segments
​
cent=np.array([int(N/2),int(N/2)])# center of primary mirror i.e. center of the center segment
a=25#radius of hexagon in pixels
gap=1#gap between segments in pixels
​
​
x_cords=np.zeros(((2*(n_ring)+1)))#will store x cordinates of each line of segment
​
####------X cordinate generation for each line of segments-------------
for i in range(int(len(x_cords)/2),len(x_cords)):
    x_cords[i]=(a+gap)*(1.5)*(i-int(len(x_cords)/2))
​
for i in range(0,int(len(x_cords)/2)):
    x_cords[i]=(a+gap)*(1.5)*(i-int(len(x_cords)/2))
########################################################################
xx=np.cos(30*np.pi/180)*a# perpendicular length
x_line=np.arange(int(-len(x_cords)/2),int(len(x_cords)/2)+1,dtype=int)# number of x lines 
​
x_seg=[]# x coordinates of centers of all segments
y_seg=[]# x coordinates of centers of all segments
​
######----------Calculation for y co-ordinates for eaxh x line------------------
for i in range(len(x_line)):
    
    x_line_no=x_line[i]#x line number
    n_y_seg=(2*n_ring+1)-abs(x_line_no)# number of segments in each x line 
    # print(n_y_seg)
    
    for n in range(n_y_seg):
        x_seg.append(x_cords[i])# store x coordinates of all the segments in this x line
    
    if (x_line_no%2)==0:# if line no is even
        y_line=np.arange(int(-(n_y_seg)/2),int((n_y_seg)/2)+1,dtype=int)
        for y in range(len(y_line)):
            y_seg.append(2*((y_line[y]))*(xx+gap))# stores y coordinates for each segment in this x line
        # print(y_line)
        
    else:
        y_line=np.arange(int(-(n_y_seg)/2),int((n_y_seg)/2)+1,dtype=int)
        y_line=np.delete(y_line,int(len(y_line)/2))
        for y in range(len(y_line)):
            
            y_seg.append((2*abs(y_line[y])-1)*(xx+gap)*np.sign(y_line[y]))
        # print(x_line_no)
        # print(y_line)
        
            
​
mp.figure()
mp.scatter(np.array(x_seg),np.array(y_seg))# shows centers of all segments
​
​
aper=np.zeros((N,N),dtype=complex)# aperture
​
#######--------- This part of code generates the value that the center of the center segment will give for the 6 lines of the hexagon, for all the points that lie inside the hexagon, their solutions to these 6 lines must have the same sign as that for the center of the segment--------
x=cent[0]-N/2
y=cent[1]-N/2
l1_cent=y-(np.sin(60*np.pi/180)/(np.cos(60*np.pi/180)-1)*(x-a))
l2_cent=y-a*(np.sin(60*np.pi/180))
l3_cent=y-((np.sin(60*np.pi/180)/(1-np.cos(60*np.pi/180)))*(x+a))
l4_cent=y+((np.sin(60*np.pi/180)/(1-np.cos(60*np.pi/180)))*(x+a))
l5_cent=y+a*(np.sin(60*np.pi/180))
l6_cent=y+((np.sin(60*np.pi/180)/(np.cos(60*np.pi/180)-1))*(x-a))
​
lam=632*10**-9#in m
# piston=np.linspace(0,lam/4,7)
k=np.pi*2/lam
​
# tilt=lam/5
# theta=np.arctan(tilt/a)
####------------Generation of segmented aperture-------------------
for seg in range(n_seg):
    seg_cent=[]
    seg_cent.append(x_seg[seg]+int(N/2))#x center of each segment
    seg_cent.append(y_seg[seg]+int(N/2))#y center of each segment
    for i in range(int(seg_cent[0]-1*a),int(seg_cent[0]+1*a)):# array runs from center of segment to +-a to reduce the calculation time
        for j in range(int(seg_cent[1]-1*a),int(seg_cent[1]+1*a)):
            if aper[i,j]==0:
                x=i-seg_cent[0]
                y=j-seg_cent[1]
                l1=y-(np.sin(60*np.pi/180)/(np.cos(60*np.pi/180)-1)*(x-a))# line values for the point 
                l2=y-a*(np.sin(60*np.pi/180))
                l3=y-((np.sin(60*np.pi/180)/(1-np.cos(60*np.pi/180)))*(x+a))
                l4=y+((np.sin(60*np.pi/180)/(1-np.cos(60*np.pi/180)))*(x+a))
                l5=y+a*(np.sin(60*np.pi/180))
                l6=y+((np.sin(60*np.pi/180)/(np.cos(60*np.pi/180)-1))*(x-a))
                
                if l1_cent*l1>0 and l2_cent*l2>0 and l3_cent*l3>0 and l4_cent*l4>0 and l5_cent*l5>0 and l6_cent*l6>0 :   # compare line values with that for center of the segment, if all are same in sign assign value of 1 to the pixel   
                    aper[i,j]=1#*seg#*cmath.exp(np.sqrt(complex(-1))*(k*piston[seg]))#wavefront phase added on aperture
                    # aper[i,j]=1*cmath.exp(np.sqrt(complex(-1))*(k*(x*np.tan(theta))))#wavefront phase added on aperture
                    if abs(aper[i,j])==0:
                        aper[i,j]=0
mp.figure()
mp.imshow(abs(aper))
# mp.figure()
# mp.imshow(np.angle(aper)*lam/(np.pi*2))
