import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()


def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image_np = np.asarray(image)
    processed_image = process_image(image_np)
    image_batch = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(image_batch)
    probs = tf.nn.softmax(predictions[0]).numpy()
    top_k_indices = probs.argsort()[-top_k:][::-1]
    top_probs = probs[top_k_indices]
    top_classes = [str(i) for i in top_k_indices]
    return top_probs, top_classes


def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model.')

    parser.add_argument('image_path', help='Path to the image.')
    parser.add_argument('model_path', help='Path to the saved model (.h5).')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping labels to flower names.')

    args = parser.parse_args()

    #load the model
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    #predict
    probs, classes = predict(args.image_path, model, top_k=args.top_k)

    #if label_map is provided then map class indices to names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_labels = [class_names.get(cls, cls) for cls in classes]
    else:
        class_labels = classes

    #print results
    for i in range(len(probs)):
        print(f"{class_labels[i]}: {probs[i]:.4f}")


if __name__ == '__main__':
    main()
