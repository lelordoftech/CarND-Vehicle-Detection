## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar_image.png
[image2]: ./output_images/HOG_extraction.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/false_positives.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[image8]: ./output_images/combining.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the IPython notebook (in lines 01-21).
I reused this code from Udacity's lesson with a small change:
```python
block_norm='L2-Hys'

# Reuse single_img_features function inside extract_features function
def extract_features():
	file_features = single_img_features()
```

I started by reading in all the `vehicle` and `non-vehicle` images (2nd code cell of the IPython notebook).

* Number of Vehicle Image: 8792
* Number of Non Vehicle Image: 8968

Here is an example of one of each of the `vehicle` and `non-vehicle` classes (6th code cell of the IPython notebook):

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` (8th code cell of the IPython notebook):

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters:

| Parameters        | Option 1  | Option 2  | Option 3  | Option 4  |
|:-----------------:|:---------:|:---------:|:---------:|:---------:|
| HOG channel       | 0         | 1         | 1         | ALL       |
| Orient            | 6         | 6         | 9         | 9         |
| Pixels per cell   | 8         | 8         | 8         | 8         |
| Cell per block    | 2         | 2         | 2         | 2         |

And my best choice for best result (balance both of **accuracy** and **processing speed**) is option 4:

| Parameters        | Option 4  |
|:-----------------:|:---------:|
| HOG channel       | ALL       |
| Orient            | 9         |
| Pixels per cell   | 8         |
| Cell per block    | 2         |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I also selected color features to combine with HOG features:

* Binned color features: spatial_size = (32, 32)
(5th code cell of the IPython notebook in lines 23-27)
* Color histogram features: hist_bins = 32
(5th code cell of the IPython notebook in lines 31-39)

Final fucntion for extracting features is in 5th code cell of the IPython notebook in lines 111-131.

* Normalize training data before training step is in 11th code cell of the IPython notebook in line 5. I used `sklearn.preprocessing.StandardScaler`

* The code for training step is contained in the 13th code cell of the IPython notebook.
I reused this code from Udacity's lesson, ofcourse.
I trained a linear SVM using `LinearSVC`.
My laptop is so weak, so I tested with `SGDClassifier` and using `partial_fit` method to use generator. But the accuracy is seem not good as `LinearSVC`.

| Classifier    | Accuracy  |
|:-------------:|:---------:|
| LinearSVC     | 0.9895    |
| SGDClassifier | 0.9695    |

So I will continue learn how to use SGDClassifier with better parameters later.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding function is in 19th code cell of the IPython notebook in lines 05-44.
I reused this code from Udacity's lesson.

I decided to search all over the image (720, 1280, 3):

* left to right: x_start = 0 to x_stop = 1280 (I want to cover through all the width of frame)
* top to bottom: y_start = 400 to y_stop = 656 (y < 400 is toptree, sky or very small vehicles. They is too far from us; y > 656 is the front of our car. So we do not need to focus on these area)
* sub_windows = (80, 80) (1280/80=16 so we can cover through all the width of frame)
* overlap = 0.5 (I think it is enough for detection and give best result)
(20th code cell of the IPython notebook)

And came up with this:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the 22th code cell of the IPython notebook.
I reused this code from Udacity's lesson.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.
To decrease false positives, beside using threshold method I also add some conditions for bounding box as below:

* Should have ymin value should be in range (ystart : ystart + offset)
* Should have width value is more than or equal height value

```python
if (np.min(nonzeroy) < ystart+32): # check ymin
    # Define a bounding box based on min/max x and y
    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    if (bbox[1][0]-bbox[0][0] >= bbox[1][1]-bbox[0][1]): # width >= height
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
```

I have a little change is appying threshold with 50% heatmap:
```python
def pipeline_heatmap():
    threshold = max(np.concatenate(heat).ravel())*0.5 # Reduce detection result to reduce noise
    heat = apply_threshold(heat, threshold)
```

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

##### Here are ten frames and their corresponding heatmaps:

![alt text][image5]

##### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all ten frames:
![alt text][image6]

##### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Optionals:

* Extracting features alway take a lot of time. So I visualize progress of them by library `tqdm`.
(5th code cell of the IPython notebook in line 520)
* Building a Classifier model is also take a lot of time, so I decided to save it with `pickle` file.
(15th-16th code cell of the IPython notebook)
* To help my heatmap result more stable, I used library `collections.deque`, so I can sum the heatmap of a series of frame. In here I chose 10 frames.
(22th code cell of the IPython notebook in line 520 in lines 39, 42)
* To build a output frame (combine sliding image, RGB heatmap and final detection), I used library `PIL.Image`.
(25th code cell of the IPython notebook in line 520 in lines 12-28)
Here is a example combining output:

![alt text][image8]

* To generate a serial frame for testing, I used `VideoFileClip.subclip()` and `VideoFileClip.save_frame()` function.
(26th code cell of the IPython notebook)

---

### Reference:

* https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
* https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
* https://pymotw.com/2/collections/deque.html
* https://www.youtube.com/watch?v=P2zwrTM8ueA&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&index=5
* https://discussions.udacity.com/t/how-to-integrate-heatmaps-over-several-frames/239444/23

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

##### Techniques:

* Histogram of Oriented Gradients (HOG) feature extraction
* Binned color features extraction
* Color histogram features extraction
* Scalable Linear Support Vector Machine for classification (LinearSVC)
* Stochastic Gradient Descent (SGD) Classifier **(Tried but cannot get better result at this time)**
* Sliding Windows
* Multi-scale Windows **(Tried but cannot get better result at this time)**
* Hog Sub-sampling Window Search **(Tried but cannot get better result at this time)**
(19th code cell of the IPython notebook in line 48-111
Original function is in line 113-141)
* Heatmap, Threshold and label bounding boxes

##### Fail cases:

* My pipeline maybe fail in case it have to detect 'stranger' vehicle which is not in training dataset.
* Or in case have too much vehicle in same frame. My pipeline cannot category vehicle, cannot know how many vehicle in one frame.

##### Improve:

* Try other Classifier method, not just linear SVC to get not only better accuracy prediction but also processing faster in almost weak laptop
* Optimize sliding windows with Multi-scale Windows
* Optimize speed of main pipeline with Hog Sub-sampling Window Search
* Refer heatmap of current frame to the previous frame to make the result more smooth
* Implement other detected vehicle positions drawn: circles, cubes ...
* Combine with Advanced Lane detection project
