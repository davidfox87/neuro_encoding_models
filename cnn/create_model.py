from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D, MaxPooling1D, Dropout
from keras.layers import Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l1, l2
# custom R2-score metrics for keras backend
from keras import backend as K
from keras.constraints import maxnorm
from keras.layers import Bidirectional, LSTM, TimeDistributed, RepeatVector
from keras.backend import reshape


# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
	SS_res = K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res / (SS_tot + K.epsilon()))


def load_model(train_x, trained=False, weight_path='', neurons=64, dropout_rate=0.1):
	n_timesteps, n_features, n_outputs = train_x.shape[1], 1, 1

	kernel_size = n_timesteps-1
	inputs = Input(shape=[n_timesteps, n_features])
	x = (Conv1D(neurons, kernel_size=kernel_size, activation='relu',
			   name='conv1'))(inputs)
	x = (Dropout(dropout_rate))(x)
	x = (MaxPooling1D(pool_size=2, name='pool'))(x)
	x = (Flatten(name='flatten'))(x)

	x = RepeatVector(1)(x)
	x = (LSTM(200, activation='relu', return_sequences=True))(x)
	x = TimeDistributed(Dense(100, activation='relu'))(x)
	x = TimeDistributed(Dense(n_outputs))(x)

	predictions = Activation('linear')(x)
	model = Model(outputs=predictions, inputs=inputs)

	if trained:
		model.load_weights(weight_path)

	model.compile(optimizer='adam', loss='mse', metrics=[r_square, 'mse'])
	return model

