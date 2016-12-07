# Traffic_Sign_Detection

This project is to produce a model that gives the highest possible accuracy on The German Traffic Sign Recognition Benchmark. http://benchmark.ini.rub.de/?section=gtsrb&subsection=news There are 39209 training images and 12630 images in the test set. 
My convolutional neural network acchieved 98.971% accuracy rate on the Kaggle hold-out test dataset. I will briefly introdue the techniques I used to accomplish this performance. 

1. The convolutional neural network has the following architecture:
(1): nn.SpatialConvolution(3 -> 100, 7x7)
(2): nn.ReLU
(3): nn.SpatialMaxPooling(2x2, 2,2)
(4): nn.SpatialConvolution(100 -> 150, 4x4)
(5): nn.ReLU
(6): nn.SpatialMaxPooling(2x2, 2,2)
(7): nn.SpatialConvolution(150 -> 250, 4x4)
(8): nn.ReLU
(9): nn.SpatialMaxPooling(2x2, 2,2)
(10): nn.View(2250)
(11): nn.Dropout(0.500000)
(12): nn.Linear(2250 -> 300)
(13): nn.BatchNormalization
(14): nn.ReLU
(15): nn.Linear(300 -> 43)
(16): nn.LogSoftMax

2. Data Preprocessing: The actual traffic sign is not always centered within the image; its bouding box is part of the data annotations, according to which I cropped all images and process only the image within the bounding box. Since the network architecture requireds all training images to be of equal size, I first based on visual inspection resize all images to 48*48 pixels. To avoid losing information by resizing bigger images to a smaller size, I did random distortion for each image, inspired by work of (Ciresan, Meier, Masci, and Schimidhuber). The detail is +-10% of the image size for translation, 0.9 - 1.1 for scaling and +-5 degree of rotation, and the final fixed sized images is obtained using bilinear interpolation of the distorted input image. I also tried contrast normalization (Sermanet & LeCun, 2011), but it did not improve the accuracy rate and slow down the training process, so I did not include it in the final model, but it is still an option in the code. 

3. Some techniques against overfit and accelerate convergence: dropout (weights in fully connect layer has probability 0.5 to shrink to zero) (krizhevsky et al nips 2012) Batch Normalization (making normalization a part of the model architecture and performing the normalization for each training mini-batch, also allows us to use much higher learning rates and be less careful about initialization) (Sergey Ioffe, Christian Szegedy) weight decay 0.0005. During training a regularization term is added to the network's loss to compute the backprop gradient. The weight_decay value determines how dominant this regularization term will be in the gradient computation. 

4. In order to accelerate the training process, I implemented the model for GPU Computing supportive and Parallel CPU dataset iterator. I tried using 2 thread cpu reading and gpu computing enabled and the training time on hpc is approximately 56s/epoc. The final model get trained after 100 epochs. 
