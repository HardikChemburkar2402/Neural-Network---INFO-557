import math

import numpy as np
import pytest

import nn


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)


@pytest.mark.timeout(2)
def test_predict():
    input_matrix = np.array([[13, -16],
                             [-14, 17],
                             [15, -18]])
    target_binary_predictions = np.array([[0, 1],
                                          [1, 0],
                                          [0, 1]])
    Omega_0 = np.array([[.1, -.3, .5],
                        [-.2, .4, -.6]])
    Omega_1 = np.array([[-.7, .10],
                        [.8, -.11],
                        [-.9, .12]])

    # h<layer_index>_<activation_node_index><input_instance_index>
    [[h1_11, h1_12],
     [h1_21, h1_22]] = h1 = [
        [s(13 * .1 + -14 * -.3 + 15 * .5), s(-16 * .1 + 17 * -.3 - 18 * .5)],
        [s(13 * -.2 + -14 * .4 + 15 * -.6), s(-16 * -.2 + 17 * .4 + -18 * -.6)]]

    expected_raw_output = np.array([
        [s(h1_11 * -.7 + h1_21 * .10), s(h1_12 * -.7 + h1_22 * .10)],
        [s(h1_11 * .8 + h1_21 * -.11), s(h1_12 * .8 + h1_22 * -.11)],
        [s(h1_11 * -.9 + h1_21 * .12), s(h1_12 * -.9 + h1_22 * .12)]])

    net = nn.SimpleNetwork(Omega_0, Omega_1)

    model_raw_predictions = net.predict(input_matrix)
    np.testing.assert_allclose(model_raw_predictions, expected_raw_output)

    model_binary_predictions = net.predict_zero_one(input_matrix)
    np.testing.assert_array_equal(model_binary_predictions,
                                  target_binary_predictions)


@pytest.mark.timeout(2)
def test_gradients():
    input_matrix = np.array([[0, 1, 0, 1],
                             [0, 0, 1, 1]])
    target_output_matrix = np.array([[1, 0, 1, 0],
                                     [1, 1, 0, 0]])

    weights_0 = np.array([[0.1, -0.5],
                          [0.3, -0.3],
                          [0.5, -0.1]])
    weights_1 = np.array([[0.2, 0.4, 0.6],
                          [-0.2, -0.4, -0.6]])

    # f0 : pre-activation at layer 0
    # f<layer_index>_<pre-activation_node_index><input_instance_index>
    [[f0_11, f0_12, f0_13, f0_14],
     [f0_21, f0_22, f0_23, f0_24],
     [f0_31, f0_32, f0_33, f0_34]] = f0 = [
        [.1 * 0 + -.5 * 0, .1 * 1 + -.5 * 0, .1 * 0 + -.5 * 1, .1 * 1 + -.5 * 1],
        [.3 * 0 + -.3 * 0, .3 * 1 + -.3 * 0, .1 * 0 + -.3 * 1, .3 * 1 + -.3 * 1],
        [.5 * 0 + -.1 * 0, .5 * 1 + -.1 * 0, .5 * 0 + -.1 * 1, .5 * 1 + -.1 * 1]]

    # h1 : activation at layer 1
    # h<layer_index>_<activation_node_index><input_instance_index>
    [[h1_11, h1_12, h1_13, h1_14],
     [h1_21, h1_22, h1_23, h1_24],
     [h1_31, h1_32, h1_33, h1_34]] = [[s(z) for z in row] for row in f0]

    # f1 : pre-activation at layer 1
    # f<layer_index>_<pre-activation_node_index><input_instance_index>
    [[f1_11, f1_12, f1_13, f1_14],
     [f1_21, f1_22, f1_23, f1_24]] = f1 = [
        [h1_11 * .2 + h1_21 * .4 + h1_31 * .6,
         h1_12 * .2 + h1_22 * .4 + h1_32 * .6,
         h1_13 * .2 + h1_23 * .4 + h1_33 * .6,
         h1_14 * .2 + h1_24 * .4 + h1_34 * .6],
        [h1_11 * -.2 + h1_21 * -.4 + h1_31 * -.6,
         h1_12 * -.2 + h1_22 * -.4 + h1_32 * -.6,
         h1_13 * -.2 + h1_23 * -.4 + h1_33 * -.6,
         h1_14 * -.2 + h1_24 * -.4 + h1_34 * -.6]]

    # mo : model output
    # mo_<model_output_layer_node_index><input_instance_index>
    [[mo_11, mo_12, mo_13, mo_14],
     [mo_21, mo_22, mo_23, mo_24]] = [[s(x) for x in row] for row in f1]

    # dlf1 = derivative of loss w.r.t. f1 (pre-activation of model output)
    # dlf<layer_index>_<pre-activation_node_index><input_instance_index>
    [[dlf1_11, dlf1_12, dlf1_13, dlf1_14],
     [dlf1_21, dlf1_22, dlf1_23, dlf1_24]] = [
        [2 * sg(f1_11) * (mo_11 - 1),
         2 * sg(f1_12) * (mo_12 - 0),
         2 * sg(f1_13) * (mo_13 - 1),
         2 * sg(f1_14) * (mo_14 - 0)],
        [2 * sg(f1_21) * (mo_21 - 1),
         2 * sg(f1_22) * (mo_22 - 1),
         2 * sg(f1_23) * (mo_23 - 0),
         2 * sg(f1_24) * (mo_24 - 0)]]

    # grad_l1_arr = the gradient at layer 1
    # Gradient matrix is same shape as weight (Omega) matrix at layer 1
    grad_l1_arr = np.array(
        [[(dlf1_11 * h1_11 + dlf1_12 * h1_12 + dlf1_13 * h1_13 + dlf1_14 * h1_14) / 4,
          (dlf1_11 * h1_21 + dlf1_12 * h1_22 + dlf1_13 * h1_23 + dlf1_14 * h1_24) / 4,
          (dlf1_11 * h1_31 + dlf1_12 * h1_32 + dlf1_13 * h1_33 + dlf1_14 * h1_34) / 4],
         [(dlf1_21 * h1_11 + dlf1_22 * h1_12 + dlf1_23 * h1_13 + dlf1_24 * h1_14) / 4,
          (dlf1_21 * h1_21 + dlf1_22 * h1_22 + dlf1_23 * h1_23 + dlf1_24 * h1_24) / 4,
          (dlf1_21 * h1_31 + dlf1_22 * h1_32 + dlf1_23 * h1_33 + dlf1_24 * h1_34) / 4]])

    # dlf0 = derivative of loss w.r.t. f0 (pre-activation of hidden layer 1)
    # dlf<layer_index>_<activation_node_index><input_instance_index>
    [[dlf0_11, dlf0_12, dlf0_13, dlf0_14],
     [dlf0_21, dlf0_22, dlf0_23, dlf0_24],
     [dlf0_31, dlf0_32, dlf0_33, dlf0_34]] = [
        [sg(f0_11) * (.2 * dlf1_11 + -.2 * dlf1_21),
         sg(f0_12) * (.2 * dlf1_12 + -.2 * dlf1_22),
         sg(f0_13) * (.2 * dlf1_13 + -.2 * dlf1_23),
         sg(f0_14) * (.2 * dlf1_14 + -.2 * dlf1_24)],
        [sg(f0_21) * (.4 * dlf1_11 + -.4 * dlf1_21),
         sg(f0_22) * (.4 * dlf1_12 + -.4 * dlf1_22),
         sg(f0_23) * (.4 * dlf1_13 + -.4 * dlf1_23),
         sg(f0_24) * (.4 * dlf1_14 + -.4 * dlf1_24)],
        [sg(f0_31) * (.6 * dlf1_11 + -.6 * dlf1_21),
         sg(f0_32) * (.6 * dlf1_12 + -.6 * dlf1_22),
         sg(f0_33) * (.6 * dlf1_13 + -.6 * dlf1_23),
         sg(f0_34) * (.6 * dlf1_14 + -.6 * dlf1_24)]]

    # grad_l1_arr = the gradient at layer 0
    # Gradient matrix is same shape as weight (Omega) matrix at layer 0
    grad_l0_arr = np.array(
        [[(0 * dlf0_11 + 1 * dlf0_12 + 0 * dlf0_13 + 1 * dlf0_14) / 4,
          (0 * dlf0_11 + 0 * dlf0_12 + 1 * dlf0_13 + 1 * dlf0_14) / 4],
         [(0 * dlf0_21 + 1 * dlf0_22 + 0 * dlf0_23 + 1 * dlf0_24) / 4,
          (0 * dlf0_21 + 0 * dlf0_22 + 1 * dlf0_23 + 1 * dlf0_24) / 4],
         [(0 * dlf0_31 + 1 * dlf0_32 + 0 * dlf0_33 + 1 * dlf0_34) / 4,
          (0 * dlf0_31 + 0 * dlf0_32 + 1 * dlf0_33 + 1 * dlf0_34) / 4]])

    net = nn.SimpleNetwork(weights_0, weights_1)

    [input_to_hidden_gradient,
     hidden_to_output_gradient] = net.gradients(input_matrix, target_output_matrix)

    np.testing.assert_allclose(hidden_to_output_gradient, grad_l1_arr)
    np.testing.assert_allclose(input_to_hidden_gradient, grad_l0_arr)


@pytest.mark.timeout(2)
def test_train_greater_than_half():
    inputs = np.random.uniform(size=(1, 100))
    outputs = (inputs > 0.5).astype(int)

    test_inputs = np.array([[0., 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.]])
    test_outputs = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    net = nn.SimpleNetwork.random(1, 5, 5, 1)

    assert len(net.gradients(inputs, outputs)) == 3

    net.train(inputs, outputs, iterations=1000, learning_rate=1)

    assert np.count_nonzero(net.predict_zero_one(test_inputs) == test_outputs) >= 9


@pytest.mark.timeout(2)
def test_train_learning_rate():
    inputs = np.array([[0, 0, 1, 1],
                       [0, 1, 0, 1],
                       [1, 1, 1, 1]])
    outputs = np.array([[0, 1, 1, 1]])

    net = nn.SimpleNetwork.random(3, 3, 1)
    net.train(inputs, outputs, iterations=400, learning_rate=0.01)

    assert np.any(net.predict_zero_one(inputs) != outputs)

    net = nn.SimpleNetwork.random(3, 3, 1)
    net.train(inputs, outputs, iterations=400, learning_rate=1)

    assert np.all(net.predict_zero_one(inputs) == outputs)


@pytest.mark.timeout(2)
def test_train_xor():
    inputs = np.array([[0, 0, 1, 1],
                       [0, 1, 0, 1],
                       [1, 1, 1, 1]])
    outputs = np.array([[0, 1, 1, 0]])

    all_weights = [np.array([[0.0346, 0.8939, 0.5309],
                             [-0.4352, -0.5579, 0.3724],
                             [-0.6657, -0.2151, 0.2361]]),
                   np.array([[-0.2157, -1.2187, 0.9407]])]

    net = nn.SimpleNetwork(*all_weights)
    assert len(net.gradients(inputs, outputs)) == 2

    net.train(inputs, outputs, iterations=1000, learning_rate=0.5)

    assert np.all(net.predict_zero_one(inputs) == outputs)


def s(x):
    return 1 / (1 + math.exp(-x))


def sg(x):
    return s(x) * (1 - s(x))
