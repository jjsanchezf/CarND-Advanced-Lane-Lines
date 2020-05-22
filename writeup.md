**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Project_Output/undistort_output.png "Undistorted"
[image2]: ./Project_Output/test-image-undistorted.png "Road Transformed"
[image3]: ./Project_Output/binary-example_1.png "Binary Example 1"
[image4]: ./Project_Output/transform_a.png "Warp Example 1"
[image5]: ./Project_Output/fit-lane-lines.png "Fit Visual"
[image6]: ./Project_Output/example_output.png "Output"
[image7]: ./Project_Output/no_points_found.png "no Points Found"
[image8]: ./Project_Output/binary-example_2.png "Binary Example 2"
[image9]: ./Project_Output/transform_b.png "Warp Example 2"



[video1]: ./Project_Output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### This section describes the aproach used to address each of the rubric points.  

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the IPython notebook ["Camera_Calibration.ipynb"](./Camera_Calibration.ipynb).  

First of all, the number of inside corners in any given row `nx` and colum `ny` are declared. Based on these dimensions, the array `object_point_grid` is generated, such array will be the (x, y, z) coordinates of the chessboard corners in the world. It's assumed that all points lie on the same plan on the z direction,thus fixing them at `z=0`. 2 empty buffer arrays, `objpoints` and `imgpoints` are also initialized. these arrays will contain the corner coordinates of the chessboards (`objpoints`), and the actual possitions of such point on the images (`imgpoints`).

The `cv2.findChessboardCorners()` method is applied to each calibration imagas, if corners are found, `object_point_grid` is  appended to the `objpoints` buffer, and the corner possitions on the iamge to the `imgpoints` buffer.

The resulting `objpoints` and `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` method.  

Finally, the obtained coefficients are used to obtain an undistorted images using the `cv2.undistort()` Method, obtaining the following results:

![alt text][image1]

No "Chessboard Corners" where found on the following images:

![alt text][image7]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the IPython notebook ["Image_Processing.ipynb"](./Image_Processing.ipynb).  

To display a distortion corrected test image the process is simple. The method `cv2.undistort()` is applied to the test images provided using the camera matrix and distortion coefficients obtained for the previous rubric point. It is assumed that the same camera was used to take the pictures for both points. 

This resulted in the following image transformation:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is also contained in the IPython notebook ["Image_Processing.ipynb"](./Image_Processing.ipynb).  

Three diferent version of the method `generate_binary_image` where tested.

the first two methods use a  combination of color and gradient thresholds to generate the binary image. such thresholds are applied to the image on the HLS color space. both methods apply a sobel filter in the `x` direction for the `L` and `S` channels, then a threshold is applied to the scaled absolute value of the filter's result. A threshold to the `S` channel is also applied in both versions. the onli difference is the application of a threshold to the `L` channel for one of the versions.

On the third  and final version, simple color thresholds are used to generate the binary image. Such thresholding is performed on the `YCrCb` and the `LAB` color spaces. For the yellow lines a threshold is applied to the `Cr`and `Cb` channels. for the white lines the `LAB` color space is used, and a thresohold is applied to all the channels. 

unfortunately no version of the method is able to work with other than the [project_video](./project_video.mp4). Therefore, the decission of using the third version of the method was made based on the time of processing. this method is almost two time faster than the other, and since on a real life scenario there will be live video being processed, the cycle time is also important.

Here is an example of the output of the binary generation function on a test image:

![alt text][image3]
![alt text][image8]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is also contained in the IPython notebook ["Image_Processing.ipynb"](./Image_Processing.ipynb).  

For the perspective transform two versions where implemented, both versions implement a harcoded perspective transform.

The first version called `warper()` makes a transformations where the bottom part of the image stays unchanged, and the top part is then streched in order to obtain a Bird’s Eye View. This transfromation may be the more obvious, and therefore was the first approach, but as the top part is streched on space, the information remains the same, and can lead to an unsharp line on this part of the images.

![alt text][image4]

the second approach was obtain from the web page [Bird's Eye View Transformation](https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html). This approach shrinks the bottom of the image while keeping the top row unchanged. Such method preserves all available pixels from the raw image on the top where there is a lower relative resolution, it also allows us to track adjecent lines if needed.

![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is also contained in the IPython notebook ["Image_Processing.ipynb"](./Image_Processing.ipynb).  

Two different methods where implemented to satisfiy this point. 

The first method is used when no previous line is available. It uses a sliding window approach. The method first calculates an histogram of the bottom half of the iamge, given that at the bottom of the image, there is no useful information at the borders f the images  due to the warper method used,the histogram is also limited to the center of the image(`[569:711]`). from the histogram posible starting points for the left and right line are extracted and the images is then scanned with a sliding window. This ensures that  a semi-local search from window to window is performed, never straying too far from the known previous good lane line pixels. When the top of the images is reached, the pixels gathered in each window are returned as a polyfit with degree 2.

The second method is used when the probable line positions are known. this method takes as an input the left and right polynomials as a `numpy polyfit`. a `Region of Interest`is generated with the given polynomials, and a new polynomial is generated with the pixels found on such Regions, if no polynomial is found, then the first method called.

The output from this part of the pipeline is shown below.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is also contained in the IPython notebook ["Image_Processing.ipynb"](./Image_Processing.ipynb).  

First,  the scale factor from pixel units to the physical world is define, Then the method `radius_of_curvature` calculates the radius of curvature for a second degree polynomial at a given Point. since there are two polynomials (left and right lane lines), the radius of curvature is calculated for both of them at the point closest to the vehicle, and at a later point the smallest fo it its chosen as the actual radius.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is also contained in the IPython notebook ["Image_Processing.ipynb"](./Image_Processing.ipynb). 

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's the [link to the video result]( ./Project_Output/project_output_v3.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem faced was the change of light on the videos. the frames with high contrast make it hard to detect the white lines properly. For such reason the pipe line does not work on the extra videos provided. In contrast, the yellow lines are detected with less difficulty when there a high contrast exist.

I believe that when working with vision sistems one of the biggest problems is the calibration of sensors to work on different light conditions.

for the extra videos most of the time the pipe lines was able to detect at least one line, therefore a method to predict the position of the missing line, provided that the line found is a good aproximation to the one found on the previos frame, could be implemented to make the pipe line more robust.

Also I do not check for the parallelism of the lines, a point I belive to be needed.