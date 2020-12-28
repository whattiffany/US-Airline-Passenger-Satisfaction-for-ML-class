from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
def nn_model(X_train,y_train,X_test,y_test):
    model_nn = Sequential()
    model_nn.add(Dense(23, activation='relu', input_shape=(23,)))
    model_nn.add(Dense(23, activation='relu'))
    model_nn.add(Dense(1, activation='sigmoid'))
    model_nn.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    model_nn.save('model.h5')

    history = model_nn.fit(X_train, y_train, epochs=50,batch_size=128)
    # import matplotlib.pyplot as plt
    # plt.subplots_adjust(left=0,right=1.5)
    # plt.subplot(1,2,1) 
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.plot(history.history['acc'])
    # plt.subplot(1,2,2) 
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.plot(history.history['loss'])
    # plt.show()

    # history_dict = history.history
    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']

    # epochs = range(1, len(loss_values) + 1)

    # plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.show()

    evaluate = model_nn.evaluate(X_test, y_test)
    print("Evaluate:",evaluate) #顯示訓練成果
    return evaluate