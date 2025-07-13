import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
class AttentionNetworkMultiLabel():
    def __init__(self, input_shape, output_shape):
        self.input_layer = tf.keras.Input(shape=input_shape)

        # Attention mechanism
        self.fc_attention_layer = tf.keras.layers.Dense(1, activation=None)(self.input_layer)
        self.alignment_layer = tf.keras.layers.Reshape((input_shape[0],))(self.fc_attention_layer)
        self.softmax_layer = tf.keras.layers.Softmax()(self.alignment_layer)
        self.repeat_layer = tf.keras.layers.RepeatVector(input_shape[1])(self.softmax_layer)
        self.permute_layer = tf.keras.layers.Permute((2, 1))(self.repeat_layer)

        # Classification
        self.multiply_layer = tf.keras.layers.Multiply()([self.input_layer, self.permute_layer])
        self.lambda_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(self.multiply_layer)

        self.fc_layer = tf.keras.layers.Dense(128, activation='relu')(self.lambda_layer)
        self.output_layer = tf.keras.layers.Dense(output_shape, activation='sigmoid')(self.fc_layer)  # Sigmoid cho multi-label

        # Build and compile model
        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

        self.model.summary()

    def predict(self, X, p=0.5):
        predict_proba = self.model.predict(X)
        predict_binary = np.zeros_like(predict_proba, dtype=np.int32)

        for i in range(predict_proba.shape[0]):
            proba_row = predict_proba[i]
        
            above_thresh_indices = np.where(proba_row >= p)[0]

            if len(above_thresh_indices) == 0:
                # Không có nhãn nào vượt ngưỡng: chọn nhãn có xác suất cao nhất
                top_index = np.argmax(proba_row)
                predict_binary[i, top_index] = 1
            elif len(above_thresh_indices) <= 2:
                predict_binary[i, above_thresh_indices] = 1
            else:
                # Chọn 2 xác suất cao nhất trong số các nhãn vượt ngưỡng
                top2_indices = above_thresh_indices[np.argsort(proba_row[above_thresh_indices])[-2:]]
                predict_binary[i, top2_indices] = 1

        return predict_binary


    def train(self, X, y, epochs=10, batch_size=1):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def evaluate_accuracy(self, X, y_true):
        y_pred = self.predict(X)
        
        y_true = y_true.astype(np.int32)

        num_samples = y_pred.shape[0]
        correct_predictions = 0
        
        for i in range(num_samples):
            has_common_label = np.any(y_pred[i] & y_true[i])
            if has_common_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / num_samples
        return accuracy

    def evaluate_map(self, X, y_true):
        y_scores = self.model.predict(X)
        y_true = y_true.astype(np.int32)

        mAP = average_precision_score(y_true, y_scores, average='macro')
        return mAP
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        print(f"Model weights saved to: {filepath}")

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        print(f"Model weights loaded from: {filepath}")