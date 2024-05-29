from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from layered_model import define_dense_model_single_layer, define_dense_model_with_hidden_layer
from layered_model import fit_mnist_model_single_digit, evaluate_mnist_model_single_digit
from layered_model import binarize_labels, get_mnist_data

def test_define_dense_model_single_layer():
    model = define_dense_model_single_layer(43, activation_f='sigmoid', output_length=1)
    assert len(model.layers) == 1, "Model should have 1 layer"
    assert model.input_shape == (None, 43), "Input shape is not correct"
    assert model.output_shape == (None, 1), "Output shape is not correct"
    print("test_define_dense_model_single_layer passed.")

def test_define_dense_model_with_hidden_layer():
    model = define_dense_model_with_hidden_layer(43, activation_func_array=['relu', 'sigmoid'], hidden_layer_size=11, output_length=1)
    assert len(model.layers) == 2, "Model should have 2 layers"
    assert model.input_shape == (None, 43), "Input shape is not correct"
    assert model.layers[0].units == 11, "Number of units in the first layer is not correct"
    assert model.output_shape == (None, 1), "Output shape is not correct"
    print("test_define_dense_model_with_hidden_layer passed.")

def test_fit_and_predict_mnist_single_digit_one_neuron():
    model = define_dense_model_single_layer(28*28, activation_f='sigmoid', output_length=1)
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    model = fit_mnist_model_single_digit(x_train, y_train, 2, model, epochs=5, batch_size=128)
    loss, accuracy = evaluate_mnist_model_single_digit(x_test, y_test, 2, model)
    assert accuracy > 0.9, "Accuracy should be greater than 0.9 for digit 2"
    print("Test for digit 2 passed with accuracy:", accuracy)
    loss, accuracy = evaluate_mnist_model_single_digit(x_test, y_test, 3, model)
    assert accuracy < 0.9, "Accuracy should be smaller than 0.9 for digit 3"
    print("Test for digit 3 passed with accuracy:", accuracy)

if __name__ == "__main__":
    test_define_dense_model_single_layer()
    test_define_dense_model_with_hidden_layer()
    test_fit_and_predict_mnist_single_digit_one_neuron()
    print("All tests passed.")