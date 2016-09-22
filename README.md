# Convolutional-LSTM-in-Tensorflow
An implementation of convolutional lstms in tensorflow. The code is written in the same style as the `basiclstmcell` function in tensorflow and was meant to test whether this kind of implementation worked. To test this method I applied it to the bouncing ball data set created by Ilya Sutskever in this paper [Recurrent Temporal Restricted Boltzmann Machine](http://www.uoguelph.ca/~gwtaylor/publications/nips2008/rtrbm.pdf). To add velocity information I made the x and y velocities correspond to the color of the ball. This was added so I could compare the results with just next frame prediction with straight convolutions.

# Basics of how it works
All I really did was take the old lstm implementation and replace the fully connected layers with convolutional. I use the concatenated state implementation and concat on the depth dimension. I would like to redo the `rnn_cell.py` file in tensorflow with this method. This method first appears in the paper [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/pdf/1506.04214v2.pdf).

# How well does it work!
I trained two models. One with the convolutional lstm and one with straight convolutions. The files to train these are `main_conv_lstm.py` and `main_conv.py`. These will generate videos while training that show predicted sequences of length 50. The convolutional lstm model uses the last 5 frames to predict the next 4 while the convolutional model uses 1 frame to predict the next 4. This means that the convolutional lstm model has somewhat of an advantage over the convolutional so comparing these should be taken with a grain of salt. The models were trained for 200,000 steps each of batch size 16. I saw evidence that better results could be obtained with longer training times but kept them short or testing. The convolutional lstm model generated videos such as this

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/nr0lDq6uHJw/0.jpg)](https://www.youtube.com/watch?v=nr0lDq6uHJw)


The convolutional model generated videos such as this

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/3eYYdLTIoVw/0.jpg)](https://www.youtube.com/watch?v=3eYYdLTIoVw)

To accurately compare the difference between them I made a file `create_graphs.py` that shows the reconstruction error vs the future predicted step. We can see from the graph that the convolutional lstm model performs better then the convolutional model. The error was averaged over 1,200 randomly generated sequences.

![Alt text](https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow/blob/master/figs/conv_lstm_vs_conv.png "conv lstm vs conv model")





