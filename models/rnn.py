import tensorflow as tf

class RNNClassifier(object):
    def __init__(self, batch_size=10, embedding_size=1024, hidden_size=100, context_size=50):
        
        self.x = tf.keras.Input(shape=(None, embedding_size))
        h = tf.keras.layers.GRU(hidden_size, return_sequences=True)(x)
        u = tf.keras.layers.Dense(context_size, activation='tanh')(h)
        e = tf.keras.layers.Dense(1, activation=None, use_bias=False)(u)
        align_weights = tf.keras.activations.softmax(e, axis=1)
        align_weights = tf.keras.layers.Flatten()(align_weights)
        v = tf.keras.backend.batch_dot(align_weights, h, axes=[1,1])
        logits = tf.keras.layers.Dense(2, activation=None)(v)
        self.out = tf.keras.activations.softmax(logits)

    def fit(self, x_train, y_train, epochs):
        self.model = tf.keras.Model(inputs=self.x, outputs=self.out)
        # model.summary()
        opt = tf.keras.optimizers.Adam(1e-3)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        self.model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=epochs, verbose=0, validation_split=0.0, callbacks=[tf.keras.callbacks.CSVLogger('training.log')], shuffle=True)

    def predict_proba(self, x):
        return self.model.predict(x)
