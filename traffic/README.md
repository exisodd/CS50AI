# Experimentation process
Initially, I knew that I needed an input layer with shape (IMG_WIDTH, IMG_HEIGHT, 3) and
an output layer with NUM_CATEGORIES for all the different categories with softmax activation
in order to turn it into a probability distribution.

Having nowhere to go from there, I tried using the same CNN model from the lecture, with a convolutional layer with
32 filters and 3x3 kernels, a 2x2 max-pooling layer, a Flatten layer, and a dense 128
unit layer with 0.5 dropout. This didn't work out very well, so I experimented around
with different activation functions and different number of kernels and filters, but not much changed. 

I noticed that a high number of neurons per dense layer, or convolutional layer significantly slowed down 
how fast the model was learning.

The first noticeable change happened when I tried adding a second 32-filter convolution layer and 2x2 max-pooling layer.
The accuracy increased up to 94%. I experimented with different dropout rates and found out that dropping half
of the neurons was slowing down the accuracy increases in each epoch, so I opted for a 30% dropout rate instead of 50.
I noticed very high filter rates and high neurons for the dense layer didn't work out very well, so I rounded
and kept the convolution layers at 32 filter 3x3. 

After trying out different activation functions for the 2 convolutional layers, I noticed that sigmoid-sigmoid capped at
70% accuracy, relu-relu capped at 94, relu-sigmoid at 88%, and sigmoid-relu worked best, resulting in 98% accuracy.
I then toyed around with different numbers of layers again and found out that adding a second convolutional layer (relu)
immediately after the first convolutional layer (sigmoid), while doubling the amount of filters per layer and then 
max-pooling that, and then adding another convolutional/max-pool resulted in roughly 99% accuracy, 
which is the highest I got.

I also found that having 2 dense hidden layers with 64 neurons each (1 sigmoid and 1 relu activation), instead of a 
single layer with 128 units has slight but marginal improvements (about 0.02 accuracy gain)