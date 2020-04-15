import streamlit as st


import cv2
import numpy as np
import matplotlib.pyplot as plt
from contourutils.contourutils import *

import base64
import textwrap
from scipy.spatial import ConvexHull
from pathlib import Path

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)




reduction_level = st.slider("reduction level",0.0,1.5,0.01)
k = st.number_input("number of colors ",2,256,4,2)

s = st.file_uploader('select an image ')
run = False 
if s != None :
    nparr = np.fromstring(s.read(), np.uint8)
    img_np = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    st.write(img_np.shape)
    run = True 
    




svg_head = '''<?xml version="1.0" standalone="yes"?>\n
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'''
svg_end = '''</svg>'''

def k_means_segment(img,k=4):
    
    IMG_SHAPE = img.shape
    img = img.flatten()
    pixel_values = img.reshape((-1, 3)) 
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(IMG_SHAPE)
    plt.imshow(segmented_image)
    plt.show()
    
    cv2.imwrite("segmenter.jpg",segmented_image)
    return centers,labels,pixel_values

def img_dfs(img,px):
    visited = set()
    x,y = img.shape
    q = [px]
    valid = lambda pt : False if (pt[0] < 0 or pt[0] > x-1)\
         else ( False if pt[1] < 0 or pt[1] > y-1 else True)
    while q :
        current = q.pop(0)
        
        cx,cy = current
        if img[cx,cy] == 0 or current in visited:
            visited.add(current)
            continue
        else : visited.add(current)
        refs = [(cx-1,cy-1),(cx,cy-1),(cx+1,cy-1),\
                (cx-1,cy),(cx+1,cy),(cx-1,cy+1),\
                (cx,cy+1),(cx+1,cy+1)
                ]
        refs = [pt for pt in refs if valid(pt)]
        q.extend(refs)
    return visited,len(visited)

def area_of_contour(ctr):
    
    if ctr.shape[0] < 3 :
        return 0 
    try :
        hull = ConvexHull(ctr)
    
        return hull.volume
    except RuntimeError :
        return 0 
def get_contours_from_segments(img,area_thresh=0):
    x,y = img.shape
    #on_pixels = set()
    #visited = set()
    #contour_list = []
    image, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [reduce_contour(ctr,reduction_level,scans=1) for ctr in contours]
    contours = [(ctr,area_of_contour(ctr) )  for ctr in contours ]

    contours = sorted(contours,key=lambda x : x[1],reverse=True)
    

    return contours#sorted(contour_list,key = lambda x : x[1] , reverse = True)


def BGR_TO_HEX(bgr_color):
    
    bgr_color = tuple(bgr_color[::-1])
    result = '#%02x%02x%02x' % tuple(bgr_color)
    return result

def get_svg_format_paths(contours,color):
    path_element = '''<path style="fill:{fill_color}; stroke:#00000000;" d="{line_path}"/>'''
    path_elements = []
    for contour in contours :
        
        contour = np.reshape(contour,(-1,2))
        path_string = "M"
        l = len(contour)
        for i,(x,y) in enumerate(contour) :
            x,y = str(x),str(y)
            c = "L"
            if(i==l-1): c = "z"
            path_string += x + ' ' + y + c
        path_elements.append(path_element.format(fill_color=BGR_TO_HEX(color),line_path=path_string)+'\n')
    return path_elements


def vectorize(image,output_file='result.svg'):
    global k 
    IMG_SHAPE = image.shape
    f = open(output_file,'w')
    svg_content = [svg_head.format(width=image.shape[1],height=image.shape[0])]
    
    centers,labels,pixel_values = k_means_segment(image,k)
    
    result = np.reshape(centers[labels],IMG_SHAPE)
    
    masks = [ [labels==i,labels!=i] for i in range(len(centers)) ]
    
    final_paths = []
    for i,mask in enumerate(masks) :
        
        base = np.zeros_like(pixel_values,dtype='uint8')
        base[mask[0]] = 1
        base[mask[1]] = 0
        imgray = cv2.Canny(np.reshape(base,IMG_SHAPE),100,200)
        
        
        iimg = np.reshape(base,IMG_SHAPE)*255
        
        
        iimg = cv2.cvtColor(iimg,cv2.COLOR_RGB2GRAY)
        contours_with_areas = get_contours_from_segments(iimg,100)
        contours = [ctr for ctr,area in contours_with_areas]
        
        areas = [area for ctr,area in contours_with_areas]
        #print('testing color : {} , contours received : {} : with areas : {}'.format(centers[i],len(contours),areas))
        paths = list(zip(get_svg_format_paths(contours,centers[i]),areas))
        
        final_paths.extend(paths)
        
    final_paths = sorted(final_paths,key = lambda x : x[1],reverse = True )
    paths = [path for path,_ in final_paths]
    svg_content.extend(paths)
    svg_content.append(svg_end)
    f.writelines(svg_content)
    f.close()
    return centers, labels

import os 

if run :
    st.image(np.array(img_np.T[::-1]).T,caption="input image")
    vectorize(img_np)
    

    with open('result.svg','r') as f :
        st.write("output image")
        render_svg(f.read())
        st.write("file size:{:.2f} KB ".format(os.stat('result.svg').st_size/(1024)))