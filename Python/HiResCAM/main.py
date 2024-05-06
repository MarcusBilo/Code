import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# https://www.kaggle.com/code/mariuszwisniewski/hirescam-for-class-activation-map-visualization

def load_img(img_path, size=None):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img


def get_img_array(img_path, size):
    img_array = load_img(img_path, size)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_preprocessed_img(img_path, size):
    img_array = get_img_array(img_path, size)
    return tf.keras.applications.xception.preprocess_input(img_array)


def make_hirescam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output * grads
    heatmap = np.sum(heatmap, axis=-1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def resize_heatmap(img_path, heatmap):
    img = load_img(img_path)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps['jet'](np.arange(256))
    jet_heatmap = jet[heatmap, :3]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    return jet_heatmap


def superimpose_heatmap(img, heatmap, alpha=0.5):
    superimposed_img = heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    img_path = 'Bvro0YD.jpg'
    img_array = get_preprocessed_img(img_path, size=(299, 299))
    model = tf.keras.applications.xception.Xception(weights='imagenet')
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    print('Predicted:', tf.keras.applications.xception.decode_predictions(preds, top=1)[0])
    heatmap = make_hirescam_heatmap(img_array, model, 'block14_sepconv2_act')
    plt.imshow(superimpose_heatmap(load_img(img_path), resize_heatmap(img_path, heatmap)))
    plt.show()


if __name__ == "__main__":
    main()
