import numpy as np


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
        learning_rate=0.001,
        epochs=20,
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
        epochs : int
            number of epochs to run
        batch_size : int
            size to sample from training data
        eps : float
            a small number
        """
        self.eps = eps
        self.layers = []
        self.objective = ObjFunc(self.eps, objective_function)
        self.epochs = epochs
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

    def add_layer(self, output_size, func_name, dropout):
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
            )
        )

    def add_final_layer(self):
        """
        Add a the final layer
        with objective function self.objective

        Parameters
        ----------
        None
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

    def update_all_layers(self, input, batch_labels, validation=False):
        """
        calculate the objective function value for input
        This is a forward pass through all the layers
        starting with the input value and ending
        with an evaluation of the objective function.

        Parameters
        ----------
        input : (m,n) ndarray
            m rows of n independent variables
        batch_labels : (m, n') ndarray
            m rows of n' dependent variables
        validation : boolean
            If True, then calculate then predict the batch_labels 
            from the given input. In particular, do not dropout
            any ``neurons''.
        """
        final_layer = self.layers[-1]
        for layer in self.layers[:-1]:
            layer.update(input, validation=validation)
            input = layer.evaluate()
        final_layer.update(input, batch_labels, validation=validation)

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

    def run(self, all_weights=None):
        """
        Run the model.

        Parameters
        ----------
        weights : array
            either None or initial weights for each layer of the model
        """
        self.init_weights(all_weights=all_weights)
        self.training_loss = []
        self.training_acc = []
        self.val_loss = []
        self.val_acc = []
        for j in range(self.epochs):
            train_loss = 0.0
            num_acc_pred = 0.0
            for i in range(self.num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, self.num_training_samples)
                batch_labels = self.training_data_y[start:end]
                input = self.training_data_X[start:end]
                self.update_all_layers(input, batch_labels)
                train_loss += self.layers[-1].get_loss()
                num_acc_pred += self.layers[-1].get_num_acc_pred()
                self.propogate_all_layers()

            self.training_loss.append(train_loss / self.num_training_samples)
            self.training_acc.append(num_acc_pred / self.num_training_samples)
            input = self.val_data_X
            batch_labels = self.val_data_y
            self.update_all_layers(input, batch_labels)
            self.val_loss.append(
                self.layers[-1].get_loss() / self.num_val_samples
            )
            self.val_acc.append(
                self.layers[-1].get_num_acc_pred() / self.num_val_samples
            )
            self.print_results(
                epoch=j,
                train_loss=self.training_loss[-1],
                train_acc=self.training_acc[-1],
                val_loss=self.val_loss[-1],
                val_acc=self.val_acc[-1],
            )


class Layer:
    """
    Layer class
    """

    def __init__(self, shape, dropout, rng, eps, func_name="relu"):
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
            p=[1 - self.dropout, self.dropout],
        )

    def update(self, x, validation):
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
        self.input = x
        y = self.input @ self.weights + self.bias
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
        res = new_delta @ self.weights.T
        self.weights -= (
                (rate / self.batch_size) * np.dot(self.input.T, new_delta)
                )
        self.bias -= (rate / self.batch_size) * new_delta
        return res


class FinalLayer(Layer):
    """
    A class which extends the Layer class
    Includes an objective function
    """

    def __init__(self, shape, func_name, rng, obj_func, eps):
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
            shape=shape, func_name=func_name, eps=eps, dropout=0, rng=rng
        )
        self.obj_func = obj_func
        self.loss_val = 0.0
        self.num_acc_pred = 0.0
        self.input = None

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
        self.input = x
        self.batch_size = len(x)
        res = self.input @ self.weights
        self.layer_val = self.activation.evaluate(res)
        if not validation:
            self.differential = self.obj_func.differential(res, y_hat)
        self.loss_val = self.obj_func.evaluate(res, y_hat)
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
        self.weights -= (
            (rate / self.batch_size)
            * np.dot(self.input.T, self.differential)
        )
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
        return np.sum(np.dot(y_hat.T, -np.log(y, where=y > self.eps)))

class LayerWithRankOneUpdates(Layer):
    """
    A subclass of the Layer class which removes dropout
    and includes rank one weight updates
    """

    def __init__(self, shape, rng, eps, func_name="relu", num_iter=20):
        """
        Initialize the layer

        Parameters
        ----------
        shape : (int, int)
            shape[0] is the input size, and shape[1] is the output size
        rng : numpy.random.Generator
            a given random number generator
        eps : float
            a small tolerance
        func_name : {"relu", "softmax"}
            the name of the activation function for this layer
        num_iter : int
            number of times to iterate when calculating rank one updates of gradients
        """
        super().__init__(self,
                         shape=shape,
                         rng=rng,
                         eps=eps,
                         dropout=0,
                         func_name=func_name
                         )
        self.num_iter = num_iter

    def _rank_one_update(self, num_iter, mat):
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
        v = self.rng.random(mat.shape[0],1)
        w = mat.T @ v
        for _ in range(num_iter):
            v, w = A @ w, A.T @ v
            v /= np.linalg.norm(v)
            w /= np.linalg.norm(w)
        sigma = np.linalg.norm(v)
        return sigma * v * w.T






class Activation:
    """
    Activation function
    TODO: add subclasses for relu and softmax
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
        y = y - np.max(y, axis=-1)[:, None]
        res = np.exp(y)
        return np.multiply(np.reciprocal(np.sum(res, axis=-1))[:, None], res)

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
