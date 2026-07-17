import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 300
model = tf.keras.models.load_model("models/efficientnet_rust_classifier.h5")

datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
val_gen = datagen.flow_from_directory(
    "data/patches",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

preds = model.predict(val_gen)
pred_labels = (preds > 0.5).astype(int).flatten()
true_labels = val_gen.classes

print(classification_report(true_labels, pred_labels, target_names=list(val_gen.class_indices.keys())))
print(confusion_matrix(true_labels, pred_labels))