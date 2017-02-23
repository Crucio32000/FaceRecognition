# FaceRecognition
# Released under GPLv3
Face Recognition algorithm using OpenCV face recognizer. Each recognizer has its own advantages over the other, therefore using all the three and filtering them appropriately, leads to improved results respect to the unfiltered case.

For improving the accuracy, i recommend to calibrate the camera following the OpenCV Tutorial and to trim the filtering process of incoming frames depending on the camera used. GaussianBlur works reasonably well, but there are cases in which an histogram equalization is required (CLAHE is a good ready-to-go algorithm provided by OpenCV, but play with eqHist too. On the main.cpp there are routines which performs such task on all the three RGB channels or straight to the GRAY frame too).
For further clarification. feel free to contact me @ nicitanf@gmail.com
