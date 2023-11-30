import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=-1, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=-1, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    
    x1 = 432
    y1 = 348
    x2 = 600
    y2 = 420
    
    # Calculate width and height of the region of interest
    w_r = x2 - x1
    h_r = y2 - y1
    
    
    FLAGS = ng.Config('inpaint.yml')
    # USE GPU
    ng.get_gpus(0, dedicated=False)
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, h_r, w_r*2, 3))
    
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()
    t = time.time()
    for line in lines:
    # for i in range(100):
        image, mask, out = line.split()
        base = os.path.basename(mask)

        image = cv2.imread(image)
        ori_image = image.copy()
        
        mask = cv2.imread(mask)
        
        cropped_image = image[y1:y2, x1:x2, :]
        cropped_mask = mask[y1:y2, x1:x2, :]
        
        print('Shape of cropped image: {}'.format(cropped_image.shape))
        print('Shape of cropped mask: {}'.format(cropped_mask.shape))
        
        cropped_image = cv2.resize(cropped_image, (w_r, h_r))
        cropped_mask = cv2.resize(cropped_mask, (w_r, h_r))
        
        # cv2.imwrite(out, image*(1-mask/255.) + mask)
        # # continue
        # image = np.zeros((128, 256, 3))
        # mask = np.zeros((128, 256, 3))

        assert cropped_image.shape == cropped_mask.shape

        h, w, _ = cropped_image.shape
        grid = 4
        cropped_image = cropped_image[:h//grid*grid, :w//grid*grid, :]
        cropped_mask = cropped_mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(cropped_image.shape))

        cropped_image = np.expand_dims(cropped_image, 0)
        cropped_mask = np.expand_dims(cropped_mask, 0)
        
        input_image = np.concatenate([cropped_image, cropped_mask], axis=2)
        
        print(input_image.shape)
        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        
        # Blend the result with the original image with x1, y1, x2, y2
        
        ori_image[y1:y2, x1:x2, :] = result[0][:, :, ::-1]
        result = ori_image
        
        print('Processed: {}'.format(out))
        cv2.imwrite(out, result)
        

    print('Time total: {}'.format(time.time() - t))
