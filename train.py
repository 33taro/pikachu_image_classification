import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import layers


# 画像ディレクトリ
IMG_DIR = './img'
# バッチサイズ
BATCH_SIZE = 32
# VGG16を使用するため以下のサイズに設定
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = IMAGE_SIZE + (3, )

# 訓練、検証、テストの比率
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

# 実行毎に同一の結果が得られるようシード値を固定
RANDOM_STATE = 123

# 学習率
LEARNING_RATE = 0.0001

# ------------------------------------------------------------
# 1.画像とラベル取得と訓練・検証・テスト分割
# ------------------------------------------------------------
# (1) 画像読み込み
image_dataset = image_dataset_from_directory(IMG_DIR,
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMAGE_SIZE
                                             )

# (2) 画像データセットをX:画像とY:クラス名で各々配列化
image_class_names = image_dataset.class_names
img_X = []
img_Y = []
for img_ds_batch in list(image_dataset.as_numpy_iterator()):
    img_X.extend(img_ds_batch[0])
    img_Y.extend(img_ds_batch[1])

img_X = np.asarray(img_X)
img_Y = np.asarray(img_Y)

# (3) 画像データの標準化
img_X = tf.keras.applications.vgg16.preprocess_input(img_X)

# (4) データセットを訓練・検証・テストに分割
# データセットを[train, (validation + test)]に分割
img_X_train, imp_X_tmp, img_Y_train, img_Y_tmp = train_test_split(
    img_X, img_Y,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE,
    stratify=img_Y)

# (validation + test)データセットを[validation, test]に分割
VAL_TEST_SPLIT_SIZE = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
img_X_valid, img_X_test, img_Y_valid, img_Y_test = train_test_split(
    imp_X_tmp, img_Y_tmp,
    train_size=VAL_TEST_SPLIT_SIZE,
    random_state=RANDOM_STATE,
    stratify=img_Y_tmp)

# ------------------------------------------------------------
# 2.モデル構築と学習
# ------------------------------------------------------------
# (1) 転移学習のベースモデルとしてVGG16を宣言
base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                               input_shape=IMAGE_SHAPE,
                                               weights='imagenet')

# (2) 畳み込みベースの凍結
base_model.trainable = False

# (3) モデル構築
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
transfer_learning_inputs = base_model.inputs
transfer_learning_prediction = tf.keras.layers.Dense(11, activation='softmax')(x)
transfer_learning_model = tf.keras.Model(inputs=transfer_learning_inputs,
                                         outputs=transfer_learning_prediction)

# (4) モデルのコンパイル
transfer_learning_model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])



