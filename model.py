def LinearModel(input_shape=(32,32,3),activation='softmax'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),optimizer=tf.optimizers.Adam(0.0001),metrics=['accuracy'])
    return model