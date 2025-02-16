from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
from datasets import load_dataset #for downloading the Wake Vision Dataset
import tensorflow as tf #for designing and training the model 

model_name = 'wv_quality_mcunet-320kb-1mb_vww'

# Function to parse TFRecord example
def parse_tfrecord(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode image
    img = tf.image.decode_jpeg(parsed_example["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0,1]
    
    return img, parsed_example["label"]

#fixed hyperparameters 
#do not change them
input_shape = (144,144,3)
batch_size = 512
learning_rate = 0.001
epochs = 100

#load dataset
ds = load_dataset("Harvard-Edge/Wake-Vision")

#consider using also the large split
#which contains 5,760,428 images
#but be aware of its size 1.63 TB
#https://huggingface.co/datasets/Harvard-Edge/Wake-Vision-Train-Large
#ds_large = load_dataset("Harvard-Edge/Wake-Vision-Train-Large")

#correct some labels
#e.g. ds['train_quality'][0]['person'] = 1
    
train_ds = ds['train_quality'].to_tf_dataset(columns='image', label_cols='person')

val_ds = ds['validation'].to_tf_dataset(columns='image', label_cols='person')
test_ds = ds['test'].to_tf_dataset(columns='image', label_cols='person')

#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

#try your own data augmentation recipe!
data_augmentation = tf.keras.Sequential([
    data_preprocessing,
    #apply some data augmentation 
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])

raw_dataset = tf.data.TFRecordDataset("dataset_person.tfrecord")
additional_ds_person = raw_dataset.map(parse_tfrecord).map(lambda img, label: (data_preprocessing(img), label), num_parallel_calls=tf.data.AUTOTUNE)

raw_dataset = tf.data.TFRecordDataset("dataset_no_person.tfrecord")
additional_ds_no_person = raw_dataset.map(parse_tfrecord).map(lambda img, label: (data_preprocessing(img), label), num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.concatenate(additional_ds_person).concatenate(additional_ds_no_person).shuffle(100_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.AUTOTUNE)

#fixed architecture
#do not change it
inputs = keras.Input(shape=input_shape)
#
x = keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
x = keras.layers.Conv2D(16, (3,3), padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(8, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(3, 3))(x)
x = keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(160, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(160, (1,1), padding='valid')(x)
#
x = keras.layers.AveragePooling2D(5)(x)
x = keras.layers.Conv2D(2, (1,1), padding='valid')(x)
outputs = keras.layers.Reshape((2,))(x)

#model = keras.Model(inputs, outputs)

# load model from file
model = tf.keras.models.load_model(model_name + ".tf")

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

class SaveBatchOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_path="training_log.txt", batch_interval=100):
        self.file_path = file_path
        self.batch_interval = batch_interval

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if (batch + 1) % self.batch_interval == 0:  # Every 100 batches
            with open(self.file_path, "a") as f:
                f.write(f"Batch {batch + 1}:\n")
                for key, value in logs.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

# Usage Example
save_callback = SaveBatchOutputCallback("training_log.txt", batch_interval=10)

#set validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_sparse_categorical_accuracy',
    mode='max', save_best_only=True)
    
#training
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[model_checkpoint_callback, save_callback])

#Post Training Quantization (PTQ)
model = tf.keras.models.load_model(model_name + ".tf")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter(model_name + ".tflite")
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
