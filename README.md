# Convolutional-LSTM-in-Tensorflow
An implementation of convolutional lstms in tensorflow. The code is written in the same style as the basiclstmcell function in tensorflow and was meant to test whether this kind of implementation worked. To test this method I applied it to the bouncing ball data set created by Ilya Sutskever in this paper [Recurrent Temporal Restricted Boltzmann Machine](http://www.uoguelph.ca/~gwtaylor/publications/nips2008/rtrbm.pdf). To add velocity information I made the x and y velocities correspond to the color of the ball.

# Basics of how it works
All I really did was take the old lstm implementation and replace the fully connected layers with convolutional. I use the concatenated state implementation and concat on the depth dimension. I would like to redue the `rnn_cell.py` file in tensorflow with this method.

# How well does it work!
It works really well from what I have seen. I can generate fairly long videos that look pretty good. I will try to compare it to next from prediction on a straight conv network soon. The videos have a pink bloop because sometimes the generated values are more then 1 and get wrapped around to 0.

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/RjZ1VKYyHhs/0.jpg)](http://www.youtube.com/watch?v=RjZ1VKYyHhs)
