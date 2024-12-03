# run the inference of inception-v3 model with pretrained parameters on ImageNet


import tensorflow.compat.v1 as tf

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
import cv2
# import pandas as pd

module = hub.Module("./inception_v3")

# images should be resized to 299x299
input_imgs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
features = module(input_imgs)

# Provide the file indices
# spot_info = pd.read_csv('spot_info.csv', header=0, index_col=None)
# image_no = spot_info.shape[0]
# image_no = 207
# image_no = 2649
# image_no = 1776
# project_name = 'atac_merge_2x2'

# image_no = 140
# project_name = 'atac_heart_2x2'

# image_no = 315
# project_name = 'atac_heart_3x3'
# print(image_no)

# image_no = 3600
# project_name = 'slide_seq_2.2'
# print(image_no)

# image_no = 1904
# project_name = 'SpatialCITE'
# print(image_no)

# image_no = 1344
# project_name = 'skin_x2'
# print(image_no)

# image_no = 6472
# project_name = 'skin_sr_x2'
# print(image_no)

image_no = 3852
project_name = 'pdac'
print(image_no)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    img_all = np.zeros([1, 299, 299, 3])
    for scale in range(4, 5):
        print(f"----------------Scale: x{scale}----------------")
        # load all images and combine them as
        for i in range(image_no):
            # Here, all images are stored in example_img and in *.npy format
            # file_name = './example_img/' + spot_info.iloc[i, 2] + '.npy'
            # file_name = f'./{project_name}/image_x{scale}/Img_{i+1}.jpg'
            file_name = f"./{project_name}/image_x{scale}/Img_{i+1}.jpg"
            temp_ = cv2.imread(file_name)
            temp_ = cv2.resize(temp_, (299, 299))
            temp2 = temp_.astype(np.float32) / 255.0
            img_all[0, :, :, :] = temp2
            fea = sess.run(features, feed_dict={input_imgs: img_all})

            if i == 0:
                fea_all = fea
            else:
                fea_all = np.vstack((fea_all, fea))

        if (i == image_no-1):
            print('----------------Successfully load all images----------------')
        else:
            print('----------------Error for read---------------- ')
        print(fea_all.shape)
        # np.save(f'Inception_{project_name}_feature_x{scale}.npy', fea_all)
        np.save(f'./{project_name}/support_feature.npy', fea_all)