import matplotlib
matplotlib.use('TkAgg')

from functools import wraps
import  matplotlib.pyplot as plt
import pickle


def track_plot(func):
    plt.ion()
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.called += 1

        train_loss, test_loss = func(*args, **kwargs)
        if isinstance(train_loss, list):
            wrapper.train_loss += train_loss
        else:
            wrapper.train_loss.append(train_loss)

        if isinstance(test_loss, list):
            wrapper.test_loss += test_loss
        else:
            wrapper.test_loss.append(test_loss)

        plt.clf()
        x = range(len(wrapper.test_loss))

        plt.plot(x, wrapper.train_loss, 'g-')
        plt.plot(x, wrapper.test_loss, 'r-')
        plt.title('Loss')
        plt.ylim(ymax=1000)
        plt.pause(0.00001)
        plt.show(block=False)

        return train_loss, test_loss

    wrapper.called = 0
    wrapper.train_loss = []
    wrapper.test_loss = []

    wrapper.__name__ = func.__name__

    return wrapper


def draw(loss, trRSME, valRSME):
    x = range(len(loss))

    plt.plot(x, loss, 'g-')
    plt.title('Train Loss')


    plt.show()





