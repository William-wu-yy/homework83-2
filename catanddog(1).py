# 导入必要的库
import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, InceptionV3  # 导入模型
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D



# 设置随机种子
random_seed = 42
random.seed(random_seed)

# 原始数据集目录和目标目录
original_train_dir = 'D:/6222ASS/training_set/training_set'
original_test_dir = 'D:/6222ASS/test_set/test_set'
train_dir = 'D:/6222ASS/selected_data/train'
test_dir = 'D:/6222ASS/selected_data/test'

# 从原始目录获取所有图像文件路径
train_images = []
for root, dirs, files in os.walk(original_train_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            train_images.append(os.path.join(root, file))
test_images = []
for root, dirs, files in os.walk(original_test_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            test_images.append(os.path.join(root, file))

# 随机选取4000张图片作为训练集，1000张图片作为测试集
selected_train_images = random.sample(train_images, 4000)
selected_test_images = random.sample(test_images, 1000)

# 定义函数用于复制文件并保持类别结构
def copy_images(image_paths, target_dir):
    for img_path in image_paths:
        # 保持类别文件夹结构
        category_folder = os.path.basename(os.path.dirname(img_path))
        target_folder = os.path.join(target_dir, category_folder)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        shutil.copy(img_path, target_folder)

# 创建并复制训练集和测试集
copy_images(selected_train_images, train_dir)
copy_images(selected_test_images, test_dir)

# 数据预处理
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,   # 将像素值缩放到0~1之间
    shear_range=0.2,     # 随机剪切变换
    zoom_range=0.2,      # 随机缩放变换
    horizontal_flip=True # 随机水平翻转
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# 加载训练集图像数据
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 指定训练集的目录
    target_size=(224, 224),                  # 统一大小为224x224
    batch_size=32,                           # 每批次加载32张图像
    class_mode='binary'                      # 二分类任务，标签为二进制
)

# 加载验证集图像数据
validation_generator = val_datagen.flow_from_directory(
    test_dir,          # 指定验证集的目录
    target_size=(224, 224),                  # 统一大小为224x224
    batch_size=32,                           # 每批次加载32张图像
    class_mode='binary'                      # 二分类任务，标签为二进制
)


#================================================================
# 构建自定义的CNN模型
def build_cnn_model():
    model = Sequential()

    # 第一个卷积层和池化层
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第二个卷积层和池化层
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第三个卷积层和池化层
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 将多维的特征展平成一维
    model.add(Flatten())

    # 全连接层
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # 输出层，使用sigmoid激活函数，进行二分类
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model


#================================================================

def build_vgg16_model():
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


#==========================================================================
def build_resnet50_model():
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in resnet_base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(resnet_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
#==========================================================================

def build_efficientnetb0_model():
    efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in efficientnet_base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(efficientnet_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


#==========================================================================
#==========================================================================

def train_and_evaluate(model, model_name, train_gen, val_gen):
    print(f"Training {model_name} model...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=10,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        verbose=1
    )

    # 获取训练和验证准确率
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    # 打印结果
    print(f"{model_name} - Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return history  # 返回整个 history 对象

#=======================================================================
def plot_accuracy_over_epochs(history, model_name):
    # 绘制训练和验证准确率随迭代次数的变化
    plt.figure(figsize=(10, 6))

    # 绘制训练准确率
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')

    # 绘制验证准确率
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')

    plt.title(f'{model_name} Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()



# 构建模型
cnn_model = build_cnn_model()
vgg16_model = build_vgg16_model()
resnet50_model = build_resnet50_model()
efficientnetb0_model = build_efficientnetb0_model()


# 训练并获取每个模型的历史记录
cnn_history = train_and_evaluate(cnn_model, "CNN", train_generator, validation_generator)
vgg16_history = train_and_evaluate(vgg16_model, "VGG16", train_generator, validation_generator)
resnet50_history = train_and_evaluate(resnet50_model, "ResNet50", train_generator, validation_generator)
efficientnetb0_history = train_and_evaluate(efficientnetb0_model, "EfficientNetB0", train_generator, validation_generator)



# 绘制每个模型的训练准确率和验证准确率的变化折线图
plot_accuracy_over_epochs(cnn_history, "CNN")
plot_accuracy_over_epochs(vgg16_history, "VGG16")
plot_accuracy_over_epochs(resnet50_history, "ResNet50")
plot_accuracy_over_epochs(efficientnetb0_history, "EfficientNetB0")



def compare_models_accuracy_over_epochs(histories, model_names):
    plt.figure(figsize=(12, 8))

    # 遍历每个模型的 history 对象和对应的模型名称
    for history, model_name in zip(histories, model_names):
        plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')

    plt.title('Validation Accuracy Comparison over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# 收集所有模型的历史数据和名称
model_histories = [cnn_history, vgg16_history, resnet50_history, efficientnetb0_history]
model_names = ["CNN", "VGG16", "ResNet50", "EfficientNetB0"]

# 绘制所有模型的验证准确率对比图
compare_models_accuracy_over_epochs(model_histories, model_names)
# 收集所有模型的历史数据和名称
model_histories = [cnn_history, vgg16_history, resnet50_history, efficientnetb0_history]
model_names = ["CNN", "VGG16", "ResNet50", "EfficientNetB0"]

# 绘制所有模型的验证准确率对比图
compare_models_accuracy_over_epochs(model_histories, model_names)

