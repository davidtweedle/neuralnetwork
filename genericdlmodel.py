import numpy as np
from torch._lowrank import svd_lowrank
from torch import from_numpy

import tensorly as tl

tl.set_backend("numpy")

class Model:
    """
    Basic model for a neural network
    """

    def __init__(
        self,
        rng,
        training_data_X,
        training_data_y,
        val_data_X,
        val_data_y,
        objective_function="categoricalcrossentropy",
        learning_rate=0.1,
        batch_size=200,
        eps=1e-5,
    ):
        """
        Initialize the model

        Parameters
        ---------
        rng : numpy.random.Generator
            a random number generator
        training_data_X : (m,n) ndarray
            the training data
        training_data_y : (m,) ndarray
            the target labels
        val_data_X : (m',n) ndarray
            the validation data
        val_data_y : (m',) ndarray
            the validation labels
        objective_function : {'categoricalcrossentropy', 'RSS'}
            the name of the objective function
        learning_rate : float
            learning rate of the model
        batch_size : int
            size to sample from training data
        eps : float
            a small number
        """
        self.eps = eps
        self.layers = []
        self.objective = ObjFunc(self.eps, objective_function)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rng = rng
        self.training_data_X = training_data_X
        self.training_data_y = training_data_y
        self.val_data_X = val_data_X
        self.val_data_y = val_data_y
        self.input_size = training_data_X.shape[-1]
        self.num_training_samples = len(training_data_X)
        self.num_val_samples = len(val_data_X)
        self.output_size = training_data_y.shape[-1]
        self.num_batches = self.num_training_samples // self.batch_size
        if self.num_training_samples % self.batch_size != 0:
            self.num_batches += 1

    def add_layer(self, output_size, func_name, dropout, update_rule, update_args):
        """
        Add a layer to the network

        Parameters
        ----------
        output_size : int
            the dimension of the output from the layer
        func_name : {"relu", "softmax"}
            the name of the activation function
        dropout : float
            the probability to drop any particular output neuron
        update_rule : {'identity', 'rank one update', 'SVD'}
            update rule to use for updating weights
        update_args : dict
            arguments to pass to updater
        """
        if len(self.layers) > 0:
            input_size = self.layers[-1].shape[-1]
        else:
            input_size = self.input_size
        self.layers.append(
            Layer(
                shape=(input_size, output_size),
                func_name=func_name,
                dropout=dropout,
                rng=self.rng,
                eps=self.eps,
                update_rule=update_rule,
                update_args=update_args
            )
        )

    def add_final_layer(self, update_rule='identity', update_args={}):
        """
        Add a the final layer
        with objective function self.objective

        Parameters
        ----------
        update_rule : {'identity', 'rank one update', 'SVD'}
        update_args : dict
            arguments to pass to Updater constructor
        """
        if len(self.layers) > 0:
            input_size = self.layers[-1].shape[-1]
        else:
            input_size = self.input_size
        output_size = self.output_size
        func_name = None
        if self.objective.name == "categoricalcrossentropy":
            func_name = "softmax"
        elif self.objective.name == "RSS":
            func_name = "relu"
        self.layers.append(
            FinalLayer(
                shape=(input_size, output_size),
                func_name=func_name,
                obj_func=self.objective,
                rng=self.rng,
                eps=self.eps,
                update_rule=update_rule,
                update_args=update_args
            )
        )

    def init_weights(self, all_weights=None):
        """
        Initialize the weights of each layer

        Parameters
        ----------
        weights : array of length len(self.layers) or None
            if not none, contains weights to initialize each layer

        """
        if all_weights is None:
            all_weights = len(self.layers) * [None]
        for i, layer in enumerate(self.layers):
            layer.init_weights(weights=all_weights[i])

    def propogate_all_layers(self):
        """
        Calculate the gradients of all layers
        Then use back propogation to calculate
        the next gradient
        Calculation of the gradient is done from the
        final layer and each differential is passed back
        to the previous layer

        Parameters
        ----------
        None
        """
        final_layer = self.layers[-1]
        delta = final_layer.propogate(rate=self.learning_rate)
        for layer in reversed(self.layers[:-1]):
            delta = layer.propogate(rate=self.learning_rate, delta=delta)

    def update_all_layers(self, x_batch, batch_labels, validation=False):
        """
        calculate the objective function value for input
        This is a forward pass through all the layers
        starting with the input value and ending
        with an evaluation of the objective function.

        Parameters
        ----------
        x_batch : (m,n) ndarray
            m rows of n independent variables
        batch_labels : (m, n') ndarray
            m rows of n' dependent variables
        validation : boolean
            If True, then calculate then predict the batch_labels
            from the given input. In particular, do not dropout
            any ``neurons''.
        """
        final_layer = self.layers[-1]
        x = x_batch
        for layer in self.layers[:-1]:
            layer.update(x, validation=validation)
            x = layer.evaluate()
        final_layer.update(x, batch_labels, validation=validation)

    def print_results(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """
        Print the results of the current epoch

        Parameters
        ----------
        epoch : int
            the current epoch number
        train_loss : float
            the training loss of the current epoch
        train_acc : float
            the training accuracy of the current epoch
        val_loss : float
            the validation loss of the current epoch
        val_acc : float
            the validation accuracy of the current epoch
        """
        print(
            (
                f"Epoch: {epoch}\n"
                f"  Training loss:          {train_loss:.3f}\n"
                f"  Training accuracy:      {train_acc:.3f}\n"
                f"  Validation loss:        {val_loss:.3f}\n"
                f"  Validation accuracy:    {val_acc:.3f}\n"
            )
        )

    def run(self, all_weights=None, stopping_rule='epoch', epochs=20, acc=None):
        """
        Run the model.

        Parameters
        ----------
        all_weights : array
            either None or initial weights for each layer of the model
        stopping_rule : {'epoch', 'acc'}
            run for a set number of epochs if 'epoch'
            run until training accuracy is at least acc if 'acc'
        epochs : int
            number of epochs to run for if stopping_rule == 'epoch'
        acc : float
            Stop when training accuracy >= acc if stopping_rule == 'acc'
        """
        self.init_weights(all_weights=all_weights)
        self.training_loss = []
        self.training_acc = []
        self.val_loss = []
        self.val_acc = []
        epoch = 0
        while True:
            train_loss = 0.0
            num_acc_pred = 0.0
            for i in range(self.num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, self.num_training_samples)
                batch_labels = self.training_data_y[start:end]
                train_X = self.training_data_X[start:end]
                self.update_all_layers(train_X, batch_labels)
                train_loss += self.layers[-1].get_loss()
                num_acc_pred += self.layers[-1].get_num_acc_pred()
                self.propogate_all_layers()

            self.training_loss.append(train_loss / self.num_training_samples)
            self.training_acc.append(num_acc_pred / self.num_training_samples)
            val_X = self.val_data_X
            batch_labels = self.val_data_y
            self.update_all_layers(val_X, batch_labels, validation=True)
            self.val_loss.append(
                self.layers[-1].get_loss() / self.num_val_samples
            )
            self.val_acc.append(
                self.layers[-1].get_num_acc_pred() / self.num_val_samples
            )
            self.print_results(
                epoch=epoch,
                train_loss=self.training_loss[-1],
                train_acc=self.training_acc[-1],
                val_loss=self.val_loss[-1],
                val_acc=self.val_acc[-1],
            )
            if stopping_rule == 'acc' and self.training_acc[-1] > acc or epoch >= 199:
                break
            elif stopping_rule == 'epoch' and epoch >= epochs:
                break
            epoch += 1
            idx = np.arange(self.num_training_samples)
            self.rng.shuffle(idx)
            self.training_data_X = self.training_data_X[idx]
            self.training_data_y = self.training_data_y[idx]


class Layer:
    """
    Layer class
    """

    def __init__(self,
                 shape,
                 dropout,
                 rng,
                 eps,
                 func_name="relu",
                 update_rule="identity",
                 update_args={}
                 ):
        """
        Initialize the layer

        Parameters
        ----------
        shape : (int, int)
            shape[0] is the input size, and shape[1] is the output size
        dropout : float
            0 <= dropout <= 1 is the chance that any particular output node
            is set to zero on a given iteration
        rng : numpy.random.Generator
            a given random number generator
        eps : float
            a small tolerance
        func_name : {"relu", "softmax"}
            the name of the activation function for this layer
        """
        self.shape = shape
        self.eps = eps
        self.activation = Activation(eps=self.eps, name=func_name)
        self.weights = np.zeros(shape=self.shape)
        self.bias = np.zeros(shape=self.shape)
        self.rng = rng
        self.dropout = dropout
        self.dropout_weights = None
        self.layer_val = None
        self.differential = None
        self.batch_size = None
        self.x_batch = None
        self.updater = Updater(rule=update_rule,
                               **update_args
                               )

    def init_weights(self, weights=None, bias=None):
        """
        Initialize the weights

        Parameters
        ----------
        weights : (n,k) ndarray or None
           if None, then set the weights to a random initial value
           if not None then set the weights as requested
        bias : (k,) ndarray or None
        """
        if weights is None:
            weights = 0.2 * self.rng.random(self.shape) - 0.1
        if bias is None:
            bias = 0.2 * self.rng.random(self.shape[-1])
        self.weights = weights
        self.bias = bias

    def reset_dropout_weights(self):
        """
        Reset the dropout weights

        Parameters
        ----------
        None
        """
        self.dropout_weights = self.rng.choice(
            a=[0, 1],
            size=(self.batch_size, self.shape[-1]),
            p=[self.dropout, 1 - self.dropout],
        )

    def update(self, x, validation=False):
        """
        update the value of the layer for the input x
        y = R(D(xw)) is computed and stored
        in self.layer_val
        where R is the activation function
        D is the dropout layer
        w are the weights

        Parameters
        ----------
        x : (m, n) ndarray
           m input vectors of n independent variables
        validation : boolean
            if true then do not update the gradient and do not use dropout
        """
        self.batch_size = len(x)
        self.x_batch = x
        y = self.x_batch @ self.weights + self.bias
        if not validation:  # apply dropout
            self.reset_dropout_weights()
            y /= (1.0 - self.dropout)
            y = np.multiply(self.dropout_weights, y)
        self.layer_val = self.activation.evaluate(y)
        if not validation:  # compute gradients
            self.differential = self.activation.differential(y)
            self.differential = np.multiply(
                self.differential, self.dropout_weights
            )
            self.differential = self.differential.T

    def evaluate(self):
        """
        Return the most recent value of the call to
        this layers update funciton

        Parameters
        ----------
        None

        Returns
        -------
        y : (m,k) ndarray
            the result of the most recent call to
            this layers update function
        """
        return self.layer_val

    def propogate(self, rate, delta):
        """
        Update the weights based on the learning rate
        and the delta from the higher layers
        Return the resulting delta to pass onto the
        next layer.

        Parameters
        ----------
        rate : float
            learning rate given from the model
        delta : float
            gradient passed from the higher layers

        Returns
        -------
        delta' : float
            gradient of this layer

        """
        new_delta = np.multiply(delta, self.differential.T)
        low_rk_grad = self.updater.update(np.dot(self.x_batch.T, new_delta))
        res = new_delta @ self.weights.T
        self.weights -= (
            (rate / self.batch_size) * low_rk_grad
        )
        self.bias -= rate * np.mean(new_delta, axis=0)
        return res


class FinalLayer(Layer):
    """
    A class which extends the Layer class
    Includes an objective function
    """

    def __init__(self, shape, func_name, rng, obj_func, eps, update_rule='identity', update_args={}):
        """
        Initialize the final layer

        Parameters
        ----------
        shape : (int, int)
            shape[0] is the number of independent variables
            input into the layer.
            shape[1] is the number of dependent variables output by the
            layer.
        func_name : {"softmax", "relu"}
            the name of the activation function
        rng : numpy.random.Generator
            a random number generator
        obj_func : ObjFunc
            an objective function
            must be either "RSS" or "categoricalcrossentropy"
        eps : float
            a small tolerance
        """
        super().__init__(
            shape=shape, func_name=func_name, eps=eps, dropout=0, rng=rng,
            update_rule=update_rule,
            update_args=update_args
        )
        self.obj_func = obj_func
        self.loss_val = 0.0
        self.num_acc_pred = 0.0
        self.x_batch = None

    def get_loss(self):
        """
        return the most recent loss value

        Parameters
        ----------
        None

        Returns
        -------
        f(y) : float
            the most recent loss value for the current batch
        """
        return self.loss_val

    def get_num_acc_pred(self):
        """
        return the most recent number of accurate predictions

        Parameters
        ----------
        None

        Returns
        -------
        self.num_acc_pred : float
            the number of accurate predictions for the current batch
        """
        return self.num_acc_pred

    def update(self, x, y_hat, validation=False):
        """
        Update the loss value with input x and labels y_hat

        Parameters
        ----------
        x : (m,n) ndarray
            m rows of n independent variables
        y_hat : (m,k) ndarray
            m rows of k target variables
        validation : boolean
            if True then do not update the gradient
            if False, then update the gradient as usual
        """
        self.x_batch = x
        self.batch_size = len(x)
        res = self.x_batch @ self.weights + self.bias
        self.layer_val = self.activation.evaluate(res)
        if not validation:
            self.differential = self.obj_func.differential(self.layer_val, y_hat)
        self.loss_val = self.obj_func.evaluate(self.layer_val, y_hat)
        self.num_acc_pred = 1.0 * np.sum(
            np.argmax(self.layer_val, axis=-1) == np.argmax(y_hat, axis=-1)
        )

    def propogate(self, rate):
        """
        update the weights of the model based on the learning rate

        Parameters
        ----------
        rate : float
            the learning rate to use

        Returns
        -------
        delta : (m,k) ndarray
            the gradient of the current layer
        """
        res = self.differential @ self.weights.T
        low_rk_grad = self.updater.update(
            np.dot(self.x_batch.T, self.differential))
        self.weights -= (
            (rate / self.batch_size) * low_rk_grad
        )
        self.bias -= rate * np.mean(self.differential, axis=0)
        return res


class ObjFunc:
    """
    Objective function
    TODO: make this into subclasses
    """

    def __init__(self, eps, name="categoricalcrossentropy"):
        """
        Initialize the objective function with name

        Parameters
        ----------
        eps : float
            a small tolerance
        name : {"categoricalcrossentropy", "RSS"}
            the name of the objective function
        """
        self.name = name
        self.eps = eps

    def evaluate(self, y, y_hat):
        """
        Evaluate the objective function

        Parameters
        ----------
        y : (m,k) ndarray
            the result of evaluating the model at the given input
            of the current batch
        y_hat : (m,k) ndarray
            the target values of the current batch

        Returns
        -------
        f(y) : float
            the value of the objective function at y
        """
        if self.name == "categoricalcrossentropy":
            return self._crossentropy(y, y_hat)
        elif self.name == "RSS":
            return self._RSS(y, y_hat)

    def differential(self, y, y_hat):
        """
        calculate the differential at the result y
        and target y_hat

        Parameters
        ----------
        y : (m, k) ndarray
            the result of evaluating the model at the given input
            of the current batch
        y_hat : (m, k) ndarray
            the target values of the current batch

        Returns
        -------
        delta : (m,k) ndarray
            the gradient of the composition of the objective function and
            the activation function. We assume that if RSS is the objective
            function then the activation is relu and if categoricalcrossentropy
            is the objective function then the activation is softmax
        """
        return y - y_hat

    def _RSS(self, y, y_hat):
        """
        calculate the residual sum of squares of y - y_hat

        Parameters
        ----------
        y : (m, k) ndarray
            the result of evaluating the model at the given input
            of the current batch
        y_hat : (m, k) ndarray
            the target values of the current batch

        Returns
        -------
        sum (y - y_hat) ** 2 : float
        """
        res = (y - y_hat) ** 2
        return np.sum(res)

    def _crossentropy(self, y, y_hat):
        """
        Calculates the categorical crossentropy between y and y_hat

        Parameters
        ----------
        y : (m, k) ndarray
            the result of evaluating the model at the given input
            of the current batch
        y_hat : (m, k) ndarray
            the target values of the current batch

        Returns
        -------
        -log(y_i) such that y_hat_i == 1

        """
        select = y_hat == 1. 
        res = -np.sum(np.log(y[select]))
        return res


class Updater():
    """
    Updating rule for weight updates in a layer
    """

    def __init__(self, rule='identity', **kwargs):
        '''
        Construct an updater function

        Params
        ------
        rule : {'identity', 'rank one update', 'SVD', 'svd_lowrank'}
            rule to update the weights
        kwargs : dict
            keyword arguments to pass to the updater
        '''
        self.rule = rule
        self.kwargs = kwargs

    def update(self, mat):
        '''
        Update mat based on the given rule

        Params
        ------
        mat : (n,k) ndarray

        Returns
        -------
        mat' : (n,k) ndarray
            updater applied to mat
        '''
        updater = self._get_updater()
        try:
            return updater(mat, **self.kwargs)
        except np.linalg.LinAlgError as E:
            return mat

    def _get_updater(self):
        '''
        Returns an updater function according to the
        assigned rule
        '''
        if self.rule == 'identity':
            return self._identity
        elif self.rule == 'rank one update':
            return self._rank_one_update
        elif self.rule == 'SVD':
            return self._SVD
        elif self.rule == 'svd_lowrank':
            return self._svd_lowrank

    def _identity(self, mat):
        '''
        Returns the same matrix as given

        Params
        ------
        mat : (n,k) ndarray

        Returns
        -------
        mat : (n,k) ndarray
        '''
        return mat

    def _rank_one_update(self, mat, rng, num_iter=20):
        """
        Calculate a rank one approximation to as follows
        v, w <-- Aw/||Aw||, A^Tv / ||A^Tv|| and repeat >=num_iter times

        Params
        ------
        num_iter : int
            number of times to repeat the operation
        mat : (n,k) ndarray
            matrix to approximate by rank one matrix

        Returns
        -------
        sigma * v w^T : (n,k) ndarray
            a rank one approximation to mat
        """
        v = rng.random(size=(mat.shape[0], 1))
        w = mat.T @ v
        for _ in range(num_iter):
            v, w = mat @ w, mat.T @ v
            v /= np.linalg.norm(v)
            w /= np.linalg.norm(w)
        sigma = np.linalg.norm(v)
        return sigma * v * w.T

    def _SVD(self, mat, rank=3):
        '''
        Update mat with a low rank approximation
        using the singular value decomposition

        Params
        ------
        mat : (n,k) ndarray
            matrix to update
        rank : int
            rank of low rank approximation to use

        Returns
        -------
        mat' : (n,k) ndarray
            matrix of rank rank closest to mat
        '''
        u, s, vh = np.linalg.svd(mat)
        return (u[:, :rank] * s[:rank]) @ vh[:rank]

    def _svd_lowrank(self, mat, **kwargs):
        u, s, vh = tl.tenalg.svd_interface(mat, **kwargs)
        return (u * s) @ vh


class Activation:
    """
    Activation function
    """

    def __init__(self, eps, name="relu"):
        """
        Initialize the activation function with name

        Parameters
        ----------
        eps : float
            a small tolerance
        name : {"relu", "softmax"}
            the name of the activation function to use
        """
        self.name = name
        self.eps = eps

    def evaluate(self, x):
        """
        evaluate the activation function at x

        Parameters
        ----------
        x : (m, k) ndarray
            input to the activation function

        Returns
        -------
        f(x) : float
            the value of the activation function
        """
        if self.name == "relu":
            return self._relu(x)
        elif self.name == "softmax":
            return self._softmax(x)

    def differential(self, y):
        """
        calculate the differential

        Parameters
        ----------
        y : (m,k) ndarray
            input to the activation function

        Returns
        -------
        df / dy : (m,k) ndarray
            gradient at y
        """
        if self.name == "relu":
            return self._d_relu(y)
        elif self.name == "softmax":
            return self._d_softmax(y)

    def _relu(self, y):
        """
        Rectified linear unit

        Parameters
        ----------
        y : (m,k) ndarray

        Returns
        -------
        max(y,0) : (m,k) ndarray
            the Rectified Linear Unit activation at y
        """
        return np.maximum(y, 0)

    def _softmax(self, y):
        """
        Softmax of a vector

        Parameters
        ----------
        y : (m,k) ndarray

        Returns
        -------
        exp(y)/ sum(exp(y_i)) : (m,k) ndarray
            softmax of y
        """
        y = y #- np.max(y, axis=-1)[:, None]
        res = np.exp(y)
        return np.divide(res,np.sum(res, axis=-1)[:,None])

    def _d_softmax(self, y):
        """
        the derivative of softmax

        Parameters
        ----------
        y : (m,k) ndarray

        Returns
        -------
        d softmax / dy : (m,k) ndarray
            derivative of softmax
        """
        return np.dot(y, y - np.ones(shape=y.shape))

    def _d_relu(self, y):
        """
        the derivative of relu

        Parameters
        ----------
        y : (m,k) ndarray

        Returns
        -------
        d ReLU / dy : (m,k) ndarray
            derivative of relu at y
        """
        return np.greater_equal(y, 0).astype(np.float64)


def one_hot_encoding(labels, dim=10):
    one_hot_labels = labels[..., None] == np.arange(dim)[None]
    return one_hot_labels.astype(np.float64)


def test():
    import os
    import requests
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    request_opts = {"params": {"raw": "true"}}

    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    fname = "mnist.npz"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True, **request_opts)
        resp.raise_for_status()
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)
    data_file = np.load(fpath)
    x_train = data_file.get("x_train")
    x_test = data_file.get("x_test")
    y_train = data_file.get("y_train")
    y_test = data_file.get("y_test")
    data_file.close()
    x_train = x_train.reshape(len(x_train), 28 * 28)
    x_test = x_test.reshape(len(x_test), 28 * 28)
    sample_size = int(len(x_train))
    test_split = 5 * sample_size // 6
    seed = 1234
    rng = np.random.default_rng(seed=seed)
    index = rng.choice(len(x_train), sample_size)
    train_sample_X = x_train[index[:test_split]]
    val_sample_X = x_train[index[test_split:]]
    train_sample_y = y_train[index[:test_split]]
    val_sample_y = y_train[index[test_split:]]
    train_sample_X = train_sample_X * 1. / 255
    val_sample_X = val_sample_X * 1. / 255
    test_sample_X = x_test * 1. / 255
    train_sample_labels = one_hot_encoding(train_sample_y)
    val_sample_labels = one_hot_encoding(val_sample_y)
    test_sample_labels = one_hot_encoding(y_test)
    learning_rate = 0.5
    acc = 0.99
    stopping_rule = 'acc'
    pixels_per_image = 28 * 28
    num_labels = 10
    batch_size = 200
    dropout = 0
    hidden_layer_sizes = [200, 50]
    update_rule = "svd_lowrank"
    update_args = {'rank': 8, 'q': 8, 'niter': 2}
    model2 = Model(rng=rng,
                   training_data_X=train_sample_X,
                   training_data_y=train_sample_labels,
                   val_data_X=val_sample_X,
                   val_data_y=val_sample_labels,
                   objective_function="categoricalcrossentropy",
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   )
    for output_size in hidden_layer_sizes:
        model2.add_layer(output_size=output_size,
                         func_name="relu",
                         dropout=dropout,
                         update_rule=update_rule,
                         update_args=update_args,
                         )
    model2.add_final_layer(update_rule=update_rule, update_args=update_args)
    model2.run(all_weights=None, stopping_rule=stopping_rule, epochs=None, acc=acc)


if __name__ == '__main__':
    test()
