# 학습해서 Saved Model 형태로 ../model/tf_keras_cifar에 저장하자.

import tensorflow as tf
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Dropout


cifar10 = tf.keras.datasets.cifar10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0


print(f"train x\t: {train_x.shape} \ttrain y\t: {train_y.shape}")
print(f"test x\t: {test_x.shape} \ttest y\t: {test_y.shape}")


inputs = tf.keras.Input(shape=train_x.shape[1:])

upscale = tf.keras.layers.Lambda(lambda x: tf.image.resize_with_pad(x, 160, 160))(
    inputs
)


base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_tensor=upscale,
    input_shape=(160, 160, 3),
    pooling="max",
)


base_model.trainable = False

out = base_model.output
x = Flatten()(out)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])

es = tf.keras.callbacks.EarlyStopping(monitor="val_acc", verbose=1, patience=5)


model.fit(
    train_x, train_y, batch_size=128, epochs=20, callbacks=[es], validation_split=0.2
)

loss, acc = model.evaluate(test_x, test_y)
print(f"model loss: {loss:.4f} acc: {acc:.4f}")

model.save("../model/tf_keras_cifar")
