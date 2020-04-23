import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical



def test_model(model, X_val, y_val):
    preds = model.predict(X_val)
    y_pred = np.argmax(preds, axis=1)
    y_val = np.argmax(y_val, axis=1)

    print(f'Accuracy = {sum(y_val == y_pred)/ len(y_val)}')
    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, target_names=np.arange(10), title='Confusion matrix', normalize=False)


def plot_history(model):
    history = model.history
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def prepare_submission(name, model, X_test):
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    ids = np.arange(X_test.shape[0]) + 1
    df = pd.DataFrame.from_dict({'ImageId': ids, 'Label': y_pred}).set_index('ImageId')
    df.to_csv('sub_' + name + '.csv')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def read_and_prepare_data():
    im_size = 28
    num_classes = 10
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_size = len(train_df)
    test_size = len(test_df)

    y_labels = train_df['label']
    y_train = to_categorical(y_labels, num_classes)

    X_train = train_df.drop(columns='label').values.reshape((train_size, im_size, im_size, 1))
    X_test = test_df.values.reshape((test_size, im_size, im_size, 1))

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, y_train, y_labels, X_test
