# Neural Nets *from scratch*

Implementing basic neural network examples from scratch to get experience about 
lower-then-usually seen level of neural nets.

Keras & Tensorflow made us lazy... good to practice some heavy lifting sometimes :).

Here are sample results of two-class classifier net.
![Results](https://raw.githubusercontent.com/ghostFaceKillah/nn-from-scratch/master/img/one-layer-two-class-nn.png)

Here is classification error as a function of iteration. Note how ReLUs can sometimes be stuck
in suboptimal basin for quite some time.
![Classifier error](https://raw.githubusercontent.com/ghostFaceKillah/nn-from-scratch/master/img/one-layer-two-class-nn-error.png)


# What is done

- two class classifier softmax, which is _very_ flat neural network
- two class classifier neural net
- 4 class classifier neural net
- simple MNIST network

## Next steps

There is quite a bit of interesting work that can be done!!

- consider making nice ipython notebooks about this stuff
- show interesting scenario in the one layer multiclass, where RELUs saturate
  and we have to wait until they jump onto higher prob
- Write up comparison of ELU and RELU convergence, point out the effect mentioned above
- refactor layers into classes ? Can be very interesting
- implement convnet MNIST
