import tensorflow as tf

"""
   Builds a focal loss function for multi-class classification.

   alpha: A list or array of class weights (shape: [num_classes]) to address class imbalance.
   gamma: Focusing parameter for modulating factor (1 - p_t)^gamma. Default is 2.0.

   """


def multi_category_focal_loss1(alpha, gamma=2.0):
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    # alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
    # alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)
    alpha = tf.reshape(alpha, (-1, 1))

    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute p_t: if y_true=1 → use y_pred, else → use 1 - y_pred
        '''
        We want to extract the predicted probability for the correct class of each image.
        For example, if the ground truth is class 3 (one-hot vector: [0, 0, 0, 1, 0, 0, 0]) and the model predicts [0, 0, 0.1, 1, 0, 0, 0],
         it means the prediction is very accurate, since:
        The probability for the correct class (class 3) is very high (1.0)
        The probabilities for all incorrect classes are very low
        '''
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        # Standard cross-entropy: -log(p_t)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        # Compute focal loss: apply weight and class alpha, then sum over classes
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss1_fixed
