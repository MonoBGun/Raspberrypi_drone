import cv2
import numpy as np
import glob

cb_widht=10
cb_height=7
cb_square_size=29.0

criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0,0.01)


cb_3D_points=np.zeros((cb_widht*cb_height,3),np.float32)
cb_3D_points[:,:2]=np.mgrid[0:cb_widht,0:cb_height].T.reshape(-1,2)*cb_square_size

list_cb_3D_points=[]
list_cb_2D_points=[]

list_images=glob.glob('*.jpg')

for frame_name in list_images:
    img=cv2.imread(frame_name)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #find the chess board corners
    ret,corners=cv2.findChessboardCorners(gray,(7,10),None)

    #if found, add object points, image points(after refining them)
    if ret==True:
        list_cb_3D_points.append(cb_3D_points)

        corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        list_cb_2D_points.append(corners2)


        #draw and display the corners
        cv2.drawChessboardCorners(img,(cb_widht,cb_height),corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(list_cb_3D_points,list_cb_2D_points,gray.shape[::-1],None,None)

print("CalibrationMatrix :")
print(mtx)
print("Distortion :",dist)

with open('Img_Record.npy','wb') as f:
    np.save(f,mtx)
    np.save(f,dist)
