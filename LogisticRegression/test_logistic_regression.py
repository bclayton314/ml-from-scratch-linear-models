import numpy as np
import pytest
from LogisticRegression import LogisticRegression


def test_sigmoid_function_zero():
    model = LogisticRegression()
    result = model.sigmoid_function(0)
    assert np.isclose(result, 0.5)


def test_sigmoid_function_positive_and_negative():
    model = LogisticRegression()

    pos = model.sigmoid_function(10)
    neg = model.sigmoid_function(-10)

    assert pos > 0.99
    assert neg < 0.01


def test_compute_loss_small_when_predictions_are_good():
    model = LogisticRegression()

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.99, 0.01, 0.98, 0.02])
    samples = len(y_true)

    loss = model.compute_loss(y_true, y_pred, samples)

    assert loss < 0.05


def test_compute_loss_large_when_predictions_are_bad():
    model = LogisticRegression()

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.01, 0.99, 0.02, 0.98])
    samples = len(y_true)

    loss = model.compute_loss(y_true, y_pred, samples)

    assert loss > 3.0


def test_fit_initializes_weights_and_bias():
    X = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0]
    ])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, num_iterations=10)
    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None
    assert model.weights.shape == (X.shape[1], 1)


def test_fit_changes_weights_from_zero():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0]
    ])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, num_iterations=100)
    model.fit(X, y)

    assert not np.allclose(model.weights, 0)
    assert not np.isclose(model.bias, 0)


def test_predict_output_shape_and_values():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0]
    ])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (4, 1)
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_learns_simple_linearly_separable_data():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, num_iterations=5000)
    model.fit(X, y)
    preds = model.predict(X).flatten()

    accuracy = np.mean(preds == y)
    assert accuracy >= 0.95


def test_score_returns_high_accuracy_on_easy_dataset():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, num_iterations=5000)
    model.fit(X, y)
    acc = model.score(X, y)

    assert acc >= 0.95


def test_predict_on_new_points():
    X_train = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [0.5],
        [1.5],
        [3.5],
        [4.5]
    ])

    model = LogisticRegression(learning_rate=0.1, num_iterations=5000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test).flatten()

    # Expected rough boundary somewhere between 2 and 3
    assert preds[0] == 0
    assert preds[1] == 0
    assert preds[2] == 1
    assert preds[3] == 1


def test_fit_returns_self():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0]
    ])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    returned = model.fit(X, y)

    assert returned is model


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
