from Geometry import *
import numpy as np
import random
import matplotlib.pyplot as plt 

random.seed(20)

debug_mode = False

def debug_print(*args):
    if debug_mode :
        print(*args)

def reduce(contour,dist_thresh=0.1):
    contour = contour.reshape((-1,2))
    
    reduced = []

    ends = contour[::2] 
    centers = contour[1::2]
    res = [contour[0]]
    
    for i,p in enumerate(contour[:-2]): # check length
        p1 = Point(*p)
        p2 = Point(*contour[i+1])
        p3 = Point(*contour[i+2])
        L = Line(p1,p3)
        P1 = p
        P2 = contour[i+2]
        P3 = contour[i+1]
        #d = L.distanceFromPoint(p2)
        #debug_print(L,p2,d)
        d_norm = np.linalg.norm(P2-P1)
        if d_norm != 0 :
            d=abs(np.cross(P2-P1,P3-P1)/d_norm)
            #debug_print(d)
            if d > dist_thresh :
                res.append(contour[i+1])
    res.append(contour[-1])
    debug_print(len(res),len(contour))
    
    return np.array(res)
        
def reduce_contour(contour,*args, scans=10):
    for i in range(scans):
        contour = reduce(contour,*args)
    return contour 


x = np.arange(100)
y = [3+random.random()*1 for i in range(len(x))]

plt.show()
ctr = np.array(list(zip(x,y)))

if __name__ == "__main__" :
    debug_mode = True
    
    res = reduce_contour(ctr,0.1)
    
    xs = res[:,0]
    ys = res[:,1]
    
    #debug_print(list(zip(x,y)))
    #debug_print(res)

    
    plt.plot(x,y)
    plt.plot(xs,ys)
    
    plt.show()