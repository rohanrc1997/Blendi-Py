<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/blendipy.png" style="align:right" width=200px height=50px align="right">

# Blendi-Py


In essence, this project is a minimal implementation of a Pre-Trained VGG16 Deep Learning Model used for image classification. It is a Keras-based model of the 16-layer network developed and used by the VGG team in the ILSVRC-2014 competition.

The reason for using such a deep neural network is that, after training on a particular recognition task (eg. Object Recognition ),it can also be applied on another domain (eg. Bird Subcategorization) giving state-of-the-art results. This idea has powerful implications, as a model can be pre-trained and then applied on the similar required problem, which in our case, is styling a given 'base image' according to the textures(features)  of another 'reference image' to produce a new image as the output. 

What neural network does is, it tries to extract the “important points” from the both the images, that is it tries to recognize which attributes define the picture and learns from it. These learned attributes are an internal representation of the pre-trained neural network. For instance: 

<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/samples/comp/test2.jpg">

## Procedure Involved
* We first define the loss functions necessary to generate our result, namely the style loss, content loss and the total variational  loss.
* We define our optimization function, i.e. back propagation algorithm. Here we use L-BFGS because it’s faster and more efficient for smaller data.
* Then we set our style and content attributes of our model.
* Then we pass an image to our model (preferably our base image) and optimize it to minimize all the losses we defined above.

## Tested Instances
<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/samples/comp/test2.jpg">
<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/samples/comp/test1.jpg">
<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/samples/comp/test3.jpg">
<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/samples/comp/test5.jpg">
<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/samples/comp/test6.jpg">
<img src="https://github.com/rohanrc1997/Blendi-Py/blob/master/samples/comp/wgt.jpg">

### Requirements

What things you need to install first ?

* Refer to the Requirements manual for setting up the environment and required packages for the project model : [Requirements](https://github.com/rohanrc1997/Blendi-Py/blob/master/requirements) 


### Running and Executing the Blending Utility

* Before moving on forward, please ensure all the required packages are installed as mentioned in the manual above.

* Download and Extract the Project Repository to a folder. Change the current working directory to the same folder.

*Then, to build and execute the model program, write the command according to the following syntax, in general:

```
python blendi.py <base_image> <reference_image> <Number of Iterations> -style <style_weight>
```

Where :
* Base Image is the file name of the image which is to be styled according to the reference image. It should be present in the same directory as the current working directory where the program file is being built.
* Reference Image is the name of the image file whose features ( textures ) are extracted to style the base image. Again , it should also be ideally present in the same working directory as that of the program.
* Number of Iterations is a positive integer (>1) which defines the number of times L-BFGS algorithm is run to minimize our loss function. A higher value here, will produce an efficient and finer output image, at the expense of more computational time.
Hence, it is recommended  to keep it as 1 for most cases, in general.
* OPTIONAL : -style is used to specify the "style_weight" which determines the extent to which the textures from the reference image are transferred to the base image. It should be optimally kept in the range 1-20. By default, its value is set to 20 (maximum transfer of features / powerful blending ).

* Once the program finishes its execution, the final output image file will be saved in the current working directory with the name :
 "merged_iter_0.jpg"


### Sample Test Run Example

* After extracting the project folder, write the following command to run the sample test included along in the project folder itself :

```
python blendi.py base.jpg ref.jpg 1 -style 20

```
The number of iterations are set as 1 in this example.


## Built With

* [Anaconda](https://docs.continuum.io/anaconda/) - Anaconda® is a package manager, an environment manager, a Python distribution, and a collection of over 720 open source packages. It is free and easy to install, and it offers free community support.

* [Keras](https://keras.io/) - Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation.

* [Theano](http://deeplearning.net/software/theano/) - Theano is a Python library that allows you to define, optimize,
  and evaluate mathematical expressions involving multi-dimensional arrays efficiently.

* [NumPy/SciPy](https://docs.scipy.org/doc/) - NumPy/SciPy are the fundamental packages used for scientific computing with Python.

## Contributing

Contributions are always welcome! 
Please read the [contribution guidelines](https://github.com/rohanrc1997/Blendi-Py/blob/master/contribute.md) first.


## Authors

* [Rohan Chaudhary](https://github.com/rohanrc1997)

## Acknowledgments

* (https://github.com/faizankshaikh/) :  For providing the L-BFGS optimization approach for the problem



