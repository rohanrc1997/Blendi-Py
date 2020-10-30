try:
    import os
    import time
    import numpy as np
    from scipy.optimize import fmin_l_bfgs_b
    from scipy.misc import imsave, imread, imresize
    from keras import backend as K
    from keras.applications import vgg16
    from keras.preprocessing.image import load_img, img_to_array
    import sys

except:
    print( "One Or More Required Packages Not Installed Yet. Please go through the Readme.md for the requirements. \n")
    sys.exit()

root_dir = os.path.abspath('.')
base_image_path=os.path.join(root_dir,sys.argv[1])
ref_image_path=os.path.join(root_dir,sys.argv[2])

im_height=400
im_width=400

style_weight = 100
content_weight = 0.025
total_variation_weight = 1.

if len(sys.argv)==6 :
    if sys.argv[4]=="-style":
        style_weight=int(sys.argv[5])

print("\n************** Welcome to Blendi-PY ************** \n")
print("****An L-BFGS and ANN Based Image Styler/Merger**** \n")
print("Style Intensity Selected : (Default: 30) ",style_weight)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(im_height, im_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    x = x.reshape((3, im_height, im_width))
    x = x.transpose((1, 2, 0))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_image_dim_ordering('th')
base_image = K.variable(preprocess_image(base_image_path))
ref_image = K.variable(preprocess_image(ref_image_path))
final_image = K.placeholder((1, 3, im_height, im_width))

input_tensor = K.concatenate([base_image, ref_image, final_image], axis=0)

model = vgg16.VGG16(input_tensor=input_tensor,weights='imagenet',include_top=False)
print('Model Loaded')

model.summary()

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

def content_loss(base,final):
	loss = K.sum(K.square(final-base))
	print("Content loss:", loss)
	return loss

def gram_matrix(x):
	features=K.batch_flatten(x)
	gram=K.dot(features,K.transpose(features))
	print("Gram matrix: ", gram) 
	return gram

def style_loss(style,final):
	S=gram_matrix(style)
	F=gram_matrix(final)
	channels=3
	size=im_height*im_width
	return K.sum(K.square(S-F)) / (4. * (channels**2)*(size**2))

def total_variation_loss(x):
    a = K.square(x[:, :, :im_height-1, :im_width-1] - x[:, :, 1:, :im_width-1])
    b = K.square(x[:, :, :im_height-1, :im_width-1] - x[:, :, :im_height-1, 1:])

    return K.sum(K.pow(a + b, 1.25))

loss=K.variable(0.)
layer_features = outputs_dict['block4_conv2']
base_image_features= layer_features[0,:,:,:]
final_features= layer_features[2,:,:,:]
loss += content_weight * content_loss(base_image_features,final_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    final_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, final_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(final_image)


grads = K.gradients(loss, final_image)
outputs = [loss]
outputs.append(grads)
f_outputs = K.function([final_image], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, 3, im_height, im_width))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = preprocess_image(base_image_path)


for i in range(int(sys.argv[3])):
    print('Start of iteration', i)
    print("L-BFGS Optimization Running !")
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    img = deprocess_image(x.copy())
    fname = 'merged_iter_%d.jpg' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Completed in %ds' % ( end_time - start_time))



