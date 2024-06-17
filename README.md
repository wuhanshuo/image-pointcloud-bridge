# Multi-View Fusion for 3D Semantic Segmentation of Building Materials from Residential Building Facades

Hanshuo Wu's Semester Project at ETH  &lt;Multi-View Fusion for 3D Semantic Segmentation of Building Materials from Residential Building Facades>

## Files

#### MatchPointPixel.py

Import _MatchPointPixel.py_ file as a tool, to create a point cloud object and develop a map between points and pixels.  

```python
import numpy as np
import pye57
import os
from MatchPointPixel import PointCloud

PC = PointCloud(your_file_path)   # A Point Cloud Object

# Get images from the scanning
PC.image_list  # A list of six images in <numpy.ndarray> format in RGB.

# Get extrinsic and intrinsic matrix for each image
PC.transformation_matrices_list  # Length = 6, list of extrinsic matrices for six images.
PC.intrinsic_matrices_list  # Length = 6, list of intrinsic matrices for six images.

# Get point cloud raw data
PC.to_world_system()  # Return X_array, Y_array, Z_array, R_array, G_array, B_array, I_array, transformation_matrix.

# Match point to pixel
point = np.array([[0.753018],[15.486450],[4.570410],[1]], dtype = float)
PC.bridge_point_to_pixel(point)  # Return (image_index, (pixel_x, pixel_y)) and polt the result
```

#### PointCloudProcess.ipynb

Use _PointCloudProcess.ipynb_ file to  
* segment materials in the image  
* assign pixel's label to the corresponding point  
* visualize the result  

#### Quantification.ipynb

Use _Quantification.ipynb.ipynb_ file to voxelize the predict results and quantify each material.  

#### weights

The current model is YOLO v8 and the weight is saved as _best.pt_.

#### results

In _segment_ folder there are 3D segmentation results from the piepline.

_scan1-0829.npy_ is the numpy array file of the result, can be viewed in the _ResultVisualization.ipynb_ file.

## Result Overview

#### Point Cloud 1 - Image 1
<img src="https://github.com/wuhanshuo/image-pointcloud-bridge/assets/63944310/c1e1e351-7ccf-40f2-bbe5-d5ecdb5b8d2a" height="333"/>
<img src="https://github.com/wuhanshuo/Multi-View-Fusion-3D-Scene/assets/63944310/b5c63d5f-0333-4bcc-aeb7-7d413311e129" height="333"/>

#### Point Cloud 1 - Image 4
<img src="https://github.com/wuhanshuo/Multi-View-Fusion-3D-Scene/assets/63944310/a63f37df-7716-4c2a-96a4-f8191fbab269" height="333"/>
<img src="https://github.com/wuhanshuo/Multi-View-Fusion-3D-Scene/assets/63944310/1a790a2a-2494-4573-b234-38d0f02eceee" height="333"/>

## Acknowledgement
This project is supervised by Deepika Raghu, Martin Bucher and Prof. Dr. Catherine De Wolf from [the Chair of Circular Economy for Architecture](https://cea.ibi.ethz.ch/) at ETH.  
Original point cloud data are collected by Deepika Raghu, Martin Bucher and Matthew Gordon.  
The 2D segmentation model is trained on the dataset collected by Deepika Raghu. The model is YOLOv8 from [Ultralytics](https://github.com/ultralytics/ultralytics).  
The code is built with [pye57](https://github.com/davidcaron/pye57).  
Thanks for their supports!

