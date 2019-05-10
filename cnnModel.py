from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

def cnnmodel(xTrain,xTest,yTrain,yTest):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", input_shape=(48, 48, 3),
                     activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten(name="flatten"))

    model.add(Dense(4096, activation="relu",name="fc1"))
    model.add(Dense(1024, activation="relu",name="fc2"))
    model.add(Dense(512, activation="relu",name="fc3"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.summary()
    sgd = SGD(lr=5e-3, decay=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest),epochs=40, batch_size=128)
    scores = model.evaluate(xTest, yTest, verbose=0)
    print('score=', scores)

def cnn_feature():
    inputs = Input(shape=(48, 48, 3))
    model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(model)
    model = (MaxPooling2D(pool_size=(2, 2)))(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Flatten(name="flatten")(model)

    model = Dense(4096, activation="relu", name="fc1")(model)
    model = Dense(1024, activation="relu", name="fc2")(model)
    model = Dense(512, activation="relu", name="fc3")(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation="softmax")(model)
    model_cnn_asl = Model(inputs=inputs, outputs=model, name="CNN")
    model_cnn_asl.summary()
    sgd = SGD(lr=5e-3, decay=0.0005)
    model_cnn_asl.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model_output = Model(inputs=inputs,outputs=model_cnn_asl.get_layer('fc1').output)

    return model_output