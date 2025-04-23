import tensorflow as tf
from tensorflow.keras import backend as K

# FGSM (Fast Gradient Sign Method)
def generate_fgsm(model, X, y, epsilon=0.1):
    X = tf.Variable(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X)
        predictions = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, X)
    perturbation = epsilon * tf.sign(gradients)
    return X + perturbation

# PGD (Projected Gradient Descent)
def generate_pgd(model, X, y, epsilon=0.1, alpha=0.01, num_iterations=10):
    X = tf.Variable(X, dtype=tf.float32)
    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(X)
            predictions = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
        gradients = tape.gradient(loss, X)
        perturbation = alpha * tf.sign(gradients)
        X.assign_add(perturbation)
        X.assign(tf.clip_by_value(X, 0, 1))  # Project back to valid range
    return X

# AutoAttack (more sophisticated, for advanced adversarial training)
def generate_autoattack(model, X, y):
    from autoattack import AutoAttack  # AutoAttack needs to be installed
    attack = AutoAttack(model, norm='Linf', eps=0.1)
    adversarial_examples = attack.run_standard_evaluation(X, y)
    return adversarial_examples
