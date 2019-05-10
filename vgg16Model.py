from keras.layers import Dense, Dropout, Flatten, Input,Conv2D,MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

def vgg16model(xTrain,xTest,yTrain,yTest,epoch):
    inputs = Input(shape=(48, 48, 3))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Flatten(name="flatten")(x)
    model = Dense(4096, activation="relu", name="fc1")(model)
    model = Dropout(0.5)(model)
    model = Dense(4096, activation="relu", name="fc2")(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation="softmax")(model)
    model_vgg_asl = Model(inputs, model, name="vgg16")
    model_vgg_asl.summary()

    sgd = SGD(lr=5e-3, decay=0.0005)
    model_vgg_asl.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model_vgg_asl.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=epoch, batch_size=128)
    scores = model_vgg_asl.evaluate(xTest, yTest, verbose=0)
    print('score=', scores)

    return model_vgg_asl
