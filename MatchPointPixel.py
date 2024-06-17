"""
Coding: utf-8
Author: Hanshuo Wu
"""

import numpy as np
import pye57
import cv2
import matplotlib.pyplot as plt


class PointCloud(object):

    """
    A class used to represent a Point Cloud extracted from an E57 file.
    """

    def __init__(self, path , RGB = True):
        """
        Initialize the PointCloud object.
        
        Args:
            path (str): Path to the E57 file.
            RGB (bool): Whether to extract images in RGB format. Default is True.
        """

        self.file = pye57.E57(path)
        self.world_system = self.to_world_system()
        self.image_list = self.image_extract(RGB = RGB)
        self.transformation_matrices_list = self.get_transformation_matrix()
        self.intrinsic_matrices_list = self.get_intrinsic_matrix()
    
    def to_world_system(self):
        """
        Convert points from the camera system to the world system.
        Points from raw data are all in camera system, therefore translation into world system is done here.

        Returns:
            dict: A dictionary containing world coordinates and corresponding attributes.
        """
 
        matrix = np.concatenate((self.file.get_header(0).rotation_matrix , self.file.get_header(0).translation.reshape(3,1)),axis=1)
        matrix = np.concatenate((matrix , np.array([[0,0,0,1]])),axis=0)


        # Extract and concatenate point coordinates
        X = np.array([self.file.read_scan_raw(0)["cartesianX"]])   # In some files read_scan_raw(0) should be replaced with read_scan(0)
        Y = np.array([self.file.read_scan_raw(0)["cartesianY"]])
        Z = np.array([self.file.read_scan_raw(0)["cartesianZ"]])
        pts = np.concatenate((X,Y,Z , np.ones(X.shape)),axis=0)
        pts_world = matrix@pts

        R = self.file.read_scan_raw(0)["colorRed"]
        G = self.file.read_scan_raw(0)["colorGreen"]
        B = self.file.read_scan_raw(0)["colorBlue"]
        I = self.file.read_scan_raw(0)["intensity"]

        return {"pts_world":{
            "X":pts_world[0],"Y":pts_world[1],"Z":pts_world[2],
            "R":R, "G":G, "B":B, "I": I
        }, "matrix":matrix}
        
    def image_extract(self, RGB = True ):
        """
        Extract images from E57 file.
        Args:
            self.file --> E57file, <class "pye57.e57.E57">
            RGB --> default = True, if want to get images in RGB format

        Returns:
            list: A list of extracted images as <numpy.ndarray>. 
        """

        # Pictures read
        imf = self.file.image_file
        root = imf.root()
        #print("File loaded successfully!")

        if not root["images2D"]:
            print("File contains no 2D images. Exiting...")

        image_list = []
        for image_idx, image2D in enumerate(root["images2D"]):    
            # Get intrinsic matrix
            pinhole = image2D["pinholeRepresentation"]
            # Get picture from blob
            jpeg_image = pinhole["jpegImage"]
            jpeg_image_data = np.zeros(shape=jpeg_image.byteCount(), dtype=np.uint8)  # shape=count*1 array
            jpeg_image.read(jpeg_image_data, 0, jpeg_image.byteCount())
            image = cv2.imdecode(jpeg_image_data, cv2.IMREAD_COLOR)
            if RGB == True:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)
        #print(+"{0} pictures extracted.".format(len(image_list)))  
        return image_list

    
    def get_transformation_matrix(self):
        """
        Get CameraCoordinate to WorldCoordinate transformation matrix

        Args:
            self.file --> E57file, <class "pye57.e57.E57">
        Returns:
            list: A list of transformation matrices.
        """
        imf = self.file.image_file
        root = imf.root()
        transformation_matrices_list = []
        for idx, image2D in enumerate(root["images2D"]):    
            X,Y,Z = image2D["pose"]["translation"]
            X,Y,Z = X.value() , Y.value() , Z.value()
            #print("This is translation_matrix: \n",translation_matrix)
            w,x,y,z = image2D["pose"]["rotation"]   # rotation quaternion
            w = w.value()
            x = x.value()
            y = y.value()
            z = z.value()
            # From quaternion to rotation matrix
            transformation_matrix = np.linalg.inv(np.array( [[1-2*y*y-2*z*z , 2*x*y-2*z*w , 2*x*z+2*w*y , X],\
                                                             [2*x*y+2*w*z , 1-2*x*x-2*z*z , 2*y*z-2*w*x , Y],\
                                                             [2*x*z-2*w*y , 2*y*z+2*w*x , 1-2*x*x-2*y*y , Z],\
                                                             [0 , 0 , 0 , 1]],dtype = float))
            transformation_matrices_list.append(transformation_matrix)    
        #print("Get {0} transformation matrices".format(len(transformation_matrix_list)))
        #print("This is rotation_matrix: \n", rotation_matrix, end = "\n\n")
        return transformation_matrices_list

    def get_intrinsic_matrix(self):
        """
        Get camera intrinisc matrix. Generally all images in a file have the same intrinsic matrix.
        Args: 
            self.file --> E57file, <class "pye57.e57.E57">
        Returns:
            list: A list of intrinsic matrices as <numpy.ndarray>
        """
        imf = self.file.image_file
        root = imf.root()
        if not root["images2D"]:
            print("File contains no 2D images. Exiting...")    

        intrinsic_matrices_list = []

        for image_idx, image2D in enumerate(root["images2D"]):    
            # Get intrinsic matrix
            pinhole = image2D["pinholeRepresentation"]
            # Camera-to-Image
            f = - pinhole["focalLength"].value()  # Negative Z-axis towards front.
            # Image-to-Pixel
            pixel_height = pinhole["pixelHeight"].value()
            pixel_width = pinhole["pixelWidth"].value()
            principal_x = pinhole["principalPointX"].value()
            principal_y = pinhole["principalPointY"].value()
            
            intrinsic_matrix = np.array( [[f/pixel_width , 0 , principal_x,0],\
                                          [0 , f/pixel_height , principal_y,0],\
                                          [0 , 0 , 1,0]],dtype = float)
            intrinsic_matrices_list.append(intrinsic_matrix)   
        #print("Get {0} intrinsic matrices".format(len(intrinsic_matrix_list))) 

        return intrinsic_matrices_list
    
    
    def bridge_point_to_pixel(self, point, image_idx=range(6) , scale = 1):
        """
        Bridge a point to pixel.
        Args: 
            self.file --> E57file, <class "pye57.e57.E57">
            point --> given point <class "numpy.ndarray">,  shape in (4, 1). [[X],[Y],[Z],[1]]
            image_index --> <list> of intergers, the indices of images which need to go through. Use range(6) to go through all images.
            scale --> scale is not 1 only if pixel coordinate output needs a scaling transformation
            Note that original image size iss 2048x2048
        Returns:
            tuple: (index of the image, pixel coordinates of the point)
        """

        # if in_camera_system:
        #     point = (self.to_world_system["matrix"])@point
        #     #print(point)

        for i in image_idx:
            transformation_matrix = self.transformation_matrices_list[i]
            if np.dot(transformation_matrix, point)[2]>0:    # the point is in the back of this camera pose and will never exist in this image.
                #return "This point does not belong to the image!", (float("inf"),float("inf"))
                continue
                
            intrinsic_matrix  =  self.intrinsic_matrices_list[i]  # move outside
            #point2pixel_matrix =  np.dot(intrinsic_matrix, transformation_matrix)
            
            #pixel = np.dot(point2pixel_matrix, point)
            #pixel = intrinsic_matrix@transformation_matrix@(self.to_world_system)@point

            pixel = intrinsic_matrix@transformation_matrix@point
            if pixel[2][0] == 0:
                pixel[2][0] += 0.01 # To fix divided by zero error
            pixel = pixel / pixel[2][0]

            image = self.image_list[i] #height, width, channels = img.shape
            # u = round(pixel[0][0]) # nearest pixel
            # v = round(image.shape[0] - pixel[1][0])
            ## ---- PROBLEM 1 ----:  round() makes more sense, however it some times return 2048 (for points at the edge of images), which is out of dimension of image width (0,2048). Therefore use int() to avoid error.
            u = int(np.rint(pixel[0][0])) # nearest pixel 
            v = int(np.rint(image.shape[0] - pixel[1][0]))   # Pixel Coordinate to Array Index 
            #print(type(u),v)
            if 0 <= u <= image.shape[1] and 0 <= v <= image.shape[0]:
                u = np.clip(u,0,image.shape[1] - 1)     # Clip results within the frame.
                v = np.clip(v,0,image.shape[1] - 1)
                #RGB = image[u][v]      
                #plt.figure(figsize=(6, 6))
                #plt.imshow(image)
                #plt.plot([u],[v],"rx", markersize = 10)
                #plt.show()
                return i, (u*scale,v*scale)
        # if not in image 0~4: 
        # return 5, (0,0) , point
    
    

