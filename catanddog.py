# 导入必要的库
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random


# 数据预处理
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # 将像素值缩放到0~1之间
    shear_range=0.2,  # 随机剪切变换
    zoom_range=0.2,  # 随机缩放变换
    horizontal_flip=True,  # 随机水平翻转
    rotation_range=20,  # 随机旋转
    width_shift_range=0.2,  # 水平平移
    height_shift_range=0.2  # 垂直平移
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# 加载训练集和验证集数据
train_generator = train_datagen.flow_from_directory(
    'D:/迅雷下载/datasets/train',  # 指定训练集的目录
    target_size=(224, 224),  # 统一大小为224x224
    batch_size=100,  # 每批次加载32张图像
    class_mode='binary'  # 二分类任务，标签为二进制
)

validation_generator = val_datagen.flow_from_directory(
    'D:/迅雷下载/datasets/val',  # 指定验证集的目录
    target_size=(224, 224),  # 统一大小为224x224
    batch_size=32,  # 每批次加载32张图像
    class_mode='binary'  # 二分类任务，标签为二进制
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    'D:/迅雷下载/datasets/test',  # 指定测试集的目录
    target_size=(224, 224),  # 统一大小为224x224
    batch_size=1,  # 每批次加载1张图像
    class_mode=None,  # 无标签
    shuffle=False  # 不打乱顺序
)


#
#
# # 构建自定义的CNN模型
# def build_cnn_model():
#     model = Sequential()
#
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',
#                   metrics=['accuracy', 'Precision', 'Recall'])
#     return model


# 构建VGG16预训练模型
def build_vgg16_model():
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 模型训练函数
def train_and_evaluate(model, model_name, train_gen, val_gen):
    print(f"Training {model_name} model...")
    history = model.fit(
        train_gen,
        steps_per_epoch=int(np.ceil(train_gen.samples / train_gen.batch_size)),
        epochs=30,
        validation_data=val_gen,
        validation_steps=int(np.ceil(val_gen.samples / val_gen.batch_size)),
        verbose=1
    )
    # 打印训练和验证准确率
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"{model_name} - Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return history



# 可视化训练过程
def plot_accuracy_over_epochs(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.title(f'{model_name} Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# 构建模型并训练
# cnn_model = build_cnn_model()
vgg16_model = build_vgg16_model()

# cnn_history = train_and_evaluate(cnn_model, "CNN", train_generator, validation_generator)
vgg16_history = train_and_evaluate(vgg16_model, "VGG16", train_generator, validation_generator)

# 绘制准确率曲线
# plot_accuracy_over_epochs(cnn_history, "CNN")
plot_accuracy_over_epochs(vgg16_history, "VGG16")



# 获取所有测试图片的路径
test_image_paths = [os.path.join('D:/迅雷下载/datasets/test', fname) for fname in os.listdir('D:/迅雷下载/datasets/test') if fname.endswith('.jpg') or fname.endswith('.png')]

# 创建 DataFrame
test_df = pd.DataFrame({'filename': test_image_paths})

# 使用 flow_from_dataframe
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    class_mode=None,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False
)


# 加载测试集生成的预测结果
test_generator.reset()  # 重置生成器
predictions = vgg16_model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_labels = (predictions > 0.5).astype(int).ravel()

# 获取图像ID
image_ids = [os.path.basename(path) for path in test_generator.filepaths]

# 创建预测结果DataFrame
submission_df = pd.DataFrame({'ID': image_ids, 'label': predicted_labels})

# 将结果保存为CSV文件
submission_path = "D:/6222ASS/483/sampleSubmission.csv"
submission_df.to_csv(submission_path, index=False)

# 显示生成的DataFrame前5行
print(submission_df.head())

