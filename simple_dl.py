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
    ):
        """
        Initialize the model
        """
        self.layers = []
        self.objective = ObjFunc(objective_function)
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
            )
        )

    def add_final_layer(self):
        """
        Add a layer (with objective function)
        """
        if len(self.layers) > 0:
            input_size = self.layers[-1].shape[-1]
        else:
            input_size = self.input_size
        output_size = self.output_size
        if self.objective.name == "categoricalcrossentropy":
            func = Activation("softmax")
        else:
            func = Activation("relu")
        self.layers.append(
            FinalLayer(
                shape=(input_size, output_size),
                func=func,
                obj_func=self.objective,
                rng=self.rng,
            )
        )

    def init_weights(self):
        """
        Initialize the weights of each layer
        """
        for layer in self.layers:
            layer.init_weights()

    def propogate_all_layers(self):
        """
        Calculate the gradients of all layers
        Then use back propogation to calculate
        the next gradient
        """
        final_layer = self.layers[-1]
        delta = final_layer.propogate(rate=self.learning_rate)
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            delta = layer.propogate(rate=self.learning_rate, delta=delta)

    def update_all_layers(self, input, batch_labels):
        """
        calculate the objective function value for input
        TODO: disable dropout for validation
        """
        final_layer = self.layers[-1]
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            layer.update(input)
            input = layer.evaluate()
        final_layer.update(input, batch_labels)

    def print_results(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """
        Print the results of the current epoch
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

    def run(self):
        """
        Run the model
        """
        self.init_weights()
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

    def __init__(self, shape, dropout, rng, func_name="relu"):
        """
        Initialize the layer
        """
        self.shape = shape
        self.activation = Activation(name=func_name)
        self.weights = np.zeros(shape=self.shape)
        self.rng = rng
        self.dropout = dropout
        self.dropout_weights = None
        self.layer_val = None
        self.differential = None
        self.batch_size = None

    def init_weights(self):
        """
        Initialize the weights
        """
        self.weights = 0.1 * self.rng.random(self.shape)

    def reset_dropout_weights(self):
        """
        Reset the dropout weights
        TODO: add option to set dropout to zero
        """
        self.dropout_weights = self.rng.choice(
            a=[0, 1],
            size=(self.batch_size, self.shape[-1]),
            p=[1 - self.dropout, self.dropout],
        )

    def update(self, x):
        """
        update the value of the layer for the input x
        """
        self.batch_size = len(x)
        self.reset_dropout_weights()
        self.input = x
        y = self.input @ self.weights
        y = self.dropout_weights @ y
        self.layer_val = self.activation.evaluate(y)
        # TODO set self.differential

    def evalute(self):
        """
        Return the most recent value of the layer
        """
        return self.layer_val

    def propogate(self, rate, delta):
        """
        Update the weights based on the learning rate
        and the delta from the higher layers
        Return the resulting new delta for the next layer
        """
        new_delta = delta @ self.differential
        res = new_delta @ self.weights.T
        self.weights += rate * (self.input.T @ new_delta)
        return res


class FinalLayer(Layer):
    """
    A class which extends the Layer class
    Includes an objective function
    """

    def __init__(self, shape, func, rng, obj_func):
        '''
        Initialize the final layer
        '''
        super().__init__(shape=shape, func=func, dropout=0, rng=rng)
        self.obj_func = obj_func
        self.loss_val = 0.0
        self.num_acc_pred = 0.0
        self.input = None

    def get_loss(self):
        """
        return the most recent loss value
        """
        return self.loss_val

    def get_num_acc_pred(self):
        """
        return the most recent number of accurate predictions
        """
        return self.num_acc_pred

    def update(self, x, y_hat):
        """
        Update the loss value with input x and labels y_hat
        """
        self.input = x
        res = self.input @ self.weights
        self.layer_val = self.activation.evaluate(res)
        self.differential = self.obj_func.differential(res, y_hat)
        self.loss_val = self.obj_func.evaluate(res, y_hat)
        self.num_acc_pred = 1.0 * np.sum(
            np.argmax(self.layer_val, axis=-1) == np.argmax(y_hat, axis=-1)
        )

    def propogate(self, rate):
        """
        update the weights of the model based on the learning rate
        """
        res = self.differential @ self.weights.T
        self.weights += rate * self.input.T @ self.differential
        return res


class ObjFunc:
    """
    Objective functions
    """

    def __init__(self, name="categoricalcrossentropy"):
        """
        Initialize the objective function with name
        Values for name: 'categoricalcrossentropy'
                         'RSS'
        """
        self.name = name

    def evaluate(self, y, y_hat):
        """
        Evaluate the objective function
        where y is the model result
        and y_hat is the target
        """
        if self.name == "categoricalcrossentropy":
            return self._crossentropy(y, y_hat)
        if self.name == "RSS":
            return self._RSS(y, y_hat)

    def differential(self, y, y_hat):
        """
        calculate the differential at the result y
        and target y_hat
        """
        return y - y_hat

    def _RSS(self, y, y_hat):
        """
        calculate the residual sum of squares of y - y_hat
        """
        res = (y - y_hat) ** 2
        return np.sum(res)

    def _crossentropy(self, y, y_hat):
        return np.sum(np.dot(y_hat.T, -np.log(y, where=y > 0)))


class Activation:
    """
    Activation function
    TODO: add subclasses for relu and softmax
    """

    def __init__(self, name="relu"):
        """
        Initialize the activation function with name
        """
        self.name = name

    def evaluate(self, x):
        """
        evaluate the activation function at x
        """
        if self.name == "relu":
            return self._relu(x)
        elif self.name == "softmax":
            return self._softmax(x)

    def differential(self, y):
        """
        calculate the differential
        """
        if self.name == "relu":
            return self._d_relu(y)
        elif self.name == "softmax":
            return self._d_softmax(y)

    def _relu(self, y):
        """
        Rectified linear unit
        """
        return np.maximum(y, 0)

    def _softmax(self, y):
        y = y - np.max(y, axis=-1)[:, None]
        res = np.exp(y)
        return np.multiply(np.reciprocal(np.sum(res, axis=-1))[:, None], res)

    def _d_softmax(self, y):
        """
        the derivative of softmax
        """
        return np.dot(y, y - np.ones(shape=y.shape))

    def _d_relu(self, y):
        return np.greater_equal(y, 0).astype(np.float64)
