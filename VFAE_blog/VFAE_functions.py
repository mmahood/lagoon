import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# Cleaning and processing data

def violations_denoter(x):
    violations_severity = {
        'dui': 0,
        'speeding': 1,
        'stop sign/light': 2,
        'license': 3,
        'cell phone': 4,
        'paperwork': 5,
        'registration/plates': 6,
        'safe movement': 7,
        'seat belt': 8,
        'equipment': 9,
        'lights': 10,
        'truck': 11,
        'other': 12,
        'other (non-mapped)': 13
    }

    violations = []
    for k, v in violations_severity.items():
        if (k in x.lower()):
            violations.append(v)

    return min(violations)


def replace_na_categorical(df, column):
    prob = dict(df[column].value_counts() / len(df))

    keys = list(prob.keys())

    sum_prob = sum(prob.values())

    for k, v in prob.items():
        prob[k] = prob[k] / sum_prob

    prob_list = list(prob.values())

    to_fillin = np.random.choice(keys, len(df[column].loc[df[column].isnull()]), p=prob_list)

    df[column].loc[df[column].isnull()] = to_fillin

    return df


# Training model

# This runs a hyperparameter search for the selected model
def run_HPS_search(train_X, train_Y, model, n_iter, score, cv, weights=None):
    model_selection, params = get_model_and_params(model)

    RS = RandomizedSearchCV(model_selection, param_distributions=params, n_iter=n_iter, scoring=score, cv=cv,
                            error_score='0')
    RS.fit(train_X, train_Y, sample_weight=weights)
    print("Best params - ", RS.best_params_)
    print("Highest %s = %s" % (score, RS.best_score_))

    return RS.best_estimator_


# This provides the model API and hyperparameter space for the selected model
def get_model_and_params(model):
    model_selection = {
        'RFC': RandomForestClassifier(),
        'GBC': GradientBoostingClassifier(),
        'LR': LogisticRegression()
    }

    model_hyperparameters = {
        'RFC': {
            'n_estimators': range(1, 101),
            'max_depth': range(1, 50),
            'n_jobs': [-1],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None]
        },
        'GBC': {
            'loss': ['deviance', 'exponential'],
            'learning_rate': 10 ** np.linspace(-4, -1, 10),
            'n_estimators': range(50, 150)
        },
        'LR': {
            'C': 10. ** np.linspace(-3, 3, 20),
            'tol': 10 ** np.linspace(-5, -1, 20),
            'penalty': ['l2'],
            'class_weight': ['balanced']
        }
    }

    return model_selection[model], model_hyperparameters[model]


# This allows for training of a single instance of a model
def run_clf(model, train_X, train_Y, params, weights=None):
    clf = get_model_and_params(model)[0]

    clf.set_params(**params)
    clf.fit(train_X, train_Y, sample_weight=weights)

    return clf


# Check quality of model given its error metrics and discrimination parameter
# This computes error metrics and discrimination of a trained model, as well as compiles a dataframe which
# allows for visualizing the bias of the model
def check_error_and_discrimination(clf, test_X, test_Y, df_test, sensitive_features):
    Y_predict = clf.predict(test_X)

    print("F1 score = %s" % f1_score(test_Y, Y_predict))
    print("Precision score = %s" % precision_score(test_Y, Y_predict))
    print("Recall score = %s" % recall_score(test_Y, Y_predict))
    print("Accuracy score = %s" % accuracy_score(test_Y, Y_predict))

    print()

    s = np.asarray(df_test[sensitive_features])
    print("Discrimination_ratio = %s" % (discrimination(np.expand_dims(Y_predict, 1), s)))

    print()

    features = ['driver_race_Asian', 'driver_race_Black', 'driver_race_White', 'driver_race_Hispanic']

    bias = [
        features,
        list(df_test[features].mean()),
        list(df_test[features].loc[Y_predict == 1].mean()),
        list(df_test[features].loc[Y_predict == 0].mean())
    ]

    return bias


# This computes the discrimination ratio
def discrimination(y, s):
    P_a1_s1 = float(y[s == 1].sum()) / float(len(s[s == 1]))
    P_a1_s0 = float(y[s == 0].sum()) / float(len(s[s == 0]))

    disc_ratio = P_a1_s1 / P_a1_s0

    return disc_ratio

def plot_distribution(df_bias):
    iai_colors = {
        'blue': np.array([72, 196, 217]) / 255,
        'ruby': np.array([240, 86, 60]) / 255,
        'grey': np.array([49, 64, 73]) / 255,
        'beige': np.array([241, 231, 220]) / 255
    }

    plt.bar(np.arange(df_bias.shape[0])-0.2,df_bias['N_group/N_total']*100,0.2,color=iai_colors['grey'])
    plt.bar(np.arange(df_bias.shape[0]),df_bias['N_group/N_total|(predicted arrest)']*100,0.2,color=iai_colors['ruby'])
    plt.bar(np.arange(df_bias.shape[0])+0.2,df_bias['N_group/N_total|(predicted not arrest)']*100,0.2,color=iai_colors['blue'])
    plt.xticks(np.arange(df_bias.shape[0]),('Asian','Black','White','Hispanic'))
    plt.ylabel('% of people in each group')
    plt.ylim([0,100])
    plt.legend(['Overall','Predicted arrested','Predicted not arrested'])
    plt.show()

# VFAE

# This generates weights and bias variables
def gen_weights_biases(shape):
    return tf.Variable(tf.random_normal(shape, stddev=tf.sqrt(0.5 / float(shape[0]))))

# This allows the selection of an activation function
def activate(x,activation):
    if (activation == 'relu'):
        return tf.nn.relu(x)
    elif (activation == 'sigmoid'):
        return tf.nn.sigmoid(x)
    elif (activation == 'tanh'):
        return tf.nn.tanh(x)
    elif (activation == 'softmax'):
        return tf.nn.softmax(x)
    elif (activation == 'linear'):
        return x


# This returns a random and unique batch for each of the arrays in list_arrays
def get_batch(list_arrays, batch_size, index_shuffled, b):
    return [x[index_shuffled[b * batch_size:(b + 1) * batch_size]] for x in list_arrays]


def initialize_params(dims, N_epochs=1000, print_freq=100, batch_size=100, lr=1e-3, alpha=1., beta=1., D=500, gamma=1.):
    params = {
        'enc1': {
            'in_dim': dims['x'] + dims['s'],
            'hid_dim': dims['enc1_hid'],
            'out_dim': dims['z1'],
            'act': {
                'hid': 'relu',
                'mu': 'linear',
                'log_sigma': 'linear'
            }
        },
        'enc2': {
            'in_dim': dims['z1'] + 1,
            'hid_dim': dims['enc2_hid'],
            'out_dim': dims['z2'],
            'act': {
                'hid': 'relu',
                'mu': 'linear',
                'log_sigma': 'linear'
            }
        },
        'dec1': {
            'in_dim': dims['z2'] + 1,
            'hid_dim': dims['dec1_hid'],
            'out_dim': dims['z1'],
            'act': {
                'hid': 'relu',
                'mu': 'linear',
                'log_sigma': 'linear'
            }
        },
        'dec2': {
            'in_dim': dims['z1'] + dims['s'],
            'hid_dim': dims['dec2_hid'],
            'out_dim': dims['x'] + dims['s'],
            'act': {
                'hid': 'relu',
                'mu': 'sigmoid',
                'log_sigma': 'sigmoid'
            }
        },
        'us': {
            'in_dim': dims['z1'],
            'hid_dim': dims['us_hid'],
            'out_dim': dims['y_cat'],
            'act': {
                'hid': 'relu',
                'mu': 'softmax',
                'log_sigma': 'softmax'
            }
        },
        'N_epochs': N_epochs,
        'print_frequency': print_freq,
        'batch_size': batch_size,
        'lr': lr,
        'alpha': alpha,
        'beta': beta,
        'D': D,
        'gamma': gamma
    }

    return params


def initialize_weights_biases(params):
    weights = {
        'enc1': {
            'hid': gen_weights_biases([params['enc1']['in_dim'], params['enc1']['hid_dim']]),
            'mu': gen_weights_biases([params['enc1']['hid_dim'], params['enc1']['out_dim']]),
            'log_sigma': gen_weights_biases([params['enc1']['hid_dim'], params['enc1']['out_dim']])
        },
        'enc2': {
            'hid': gen_weights_biases([params['enc2']['in_dim'], params['enc2']['hid_dim']]),
            'mu': gen_weights_biases([params['enc2']['hid_dim'], params['enc2']['out_dim']]),
            'log_sigma': gen_weights_biases([params['enc2']['hid_dim'], params['enc2']['out_dim']])
        },
        'dec1': {
            'hid': gen_weights_biases([params['dec1']['in_dim'], params['dec1']['hid_dim']]),
            'mu': gen_weights_biases([params['dec1']['hid_dim'], params['dec1']['out_dim']]),
            'log_sigma': gen_weights_biases([params['dec1']['hid_dim'], params['dec1']['out_dim']])
        },
        'dec2': {
            'hid': gen_weights_biases([params['dec2']['in_dim'], params['dec2']['hid_dim']]),
            'mu': gen_weights_biases([params['dec2']['hid_dim'], params['dec2']['out_dim']]),
            'log_sigma': gen_weights_biases([params['dec2']['hid_dim'], params['dec2']['out_dim']])
        },
        'us': {
            'hid': gen_weights_biases([params['us']['in_dim'], params['us']['hid_dim']]),
            'mu': gen_weights_biases([params['us']['hid_dim'], params['us']['out_dim']]),
            'log_sigma': gen_weights_biases([params['us']['hid_dim'], params['us']['out_dim']])
        }
    }

    bias = {
        'enc1': {
            'hid': gen_weights_biases([params['enc1']['hid_dim']]),
            'mu': gen_weights_biases([params['enc1']['out_dim']]),
            'log_sigma': gen_weights_biases([params['enc1']['out_dim']])
        },
        'enc2': {
            'hid': gen_weights_biases([params['enc2']['hid_dim']]),
            'mu': gen_weights_biases([params['enc2']['out_dim']]),
            'log_sigma': gen_weights_biases([params['enc2']['out_dim']])
        },
        'dec1': {
            'hid': gen_weights_biases([params['dec1']['hid_dim']]),
            'mu': gen_weights_biases([params['dec1']['out_dim']]),
            'log_sigma': gen_weights_biases([params['dec1']['out_dim']])
        },
        'dec2': {
            'hid': gen_weights_biases([params['dec2']['hid_dim']]),
            'mu': gen_weights_biases([params['dec2']['out_dim']]),
            'log_sigma': gen_weights_biases([params['dec2']['out_dim']])
        },
        'us': {
            'hid': gen_weights_biases([params['us']['hid_dim']]),
            'mu': gen_weights_biases([params['us']['out_dim']]),
            'log_sigma': gen_weights_biases([params['us']['out_dim']])
        }
    }

    return weights, bias


def Gaussian_MLP(x_in, weights, bias, activation, epsilon):
    hidden = activate(tf.matmul(x_in, weights['hid']) + bias['hid'], activation['hid'])

    mu = activate(tf.matmul(hidden, weights['mu']) + bias['mu'], activation['mu'])

    log_sigma = activate(tf.matmul(hidden, weights['log_sigma']) + bias['log_sigma'], activation['log_sigma'])

    return mu + tf.exp(log_sigma / 2) * epsilon, mu, log_sigma

# Calculates Kullback-Leibler distance using an analytic expression, which is applicable for Guassian distributions
def KL(mu1,log_sigma_sq1,mu2=0.,log_sigma_sq2=0.):
    return 0.5*tf.reduce_sum(log_sigma_sq2-log_sigma_sq1-1+(tf.exp(log_sigma_sq1)+tf.pow(mu1-mu2,2))/tf.exp(log_sigma_sq2),axis=1)

# Calculate likelihood assuming a Gaussian distribution
def Gaussian_LH(x,mu,log_sigma):
    return 0.5 * tf.reduce_sum(np.log(2 * np.pi) + log_sigma + tf.pow(x - mu,2) / tf.exp(log_sigma), axis=1)

# Calculates fast MMD, using equation 10 of Louizos et al 2015
def fast_MMD(x1, x2, params):
    inner_difference = tf.reduce_mean(psi(x1, params), axis=0) - tf.reduce_mean(psi(x2, params), axis=0)
    return tf.tensordot(inner_difference, inner_difference, axes=1)

def psi(x, params):
    W = tf.Variable(tf.random_normal([params['enc1']['out_dim'], params['D']],
                                     stddev=tf.sqrt(0.5 / float(params['enc1']['out_dim'])),
                                     dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([params['D']], 0, 2 * np.pi, dtype=tf.float32))

    return tf.pow(2. / params['D'], 0.5) * tf.cos(tf.pow(2. / params['gamma'], 0.5) * tf.matmul(x, W) + b)

def apply_bool_mask(x, mask):
    # Apply masks
    x1 = tf.boolean_mask(x, mask)
    x2 = tf.boolean_mask(x, tf.logical_not(mask))
    # Reshape flattened output
    x1 = tf.reshape(x1, [tf.cast(tf.shape(x1)[0] / tf.shape(x)[1], tf.int32), tf.shape(x)[1]])
    x2 = tf.reshape(x2, [tf.cast(tf.shape(x2)[0] / tf.shape(x)[1], tf.int32), tf.shape(x)[1]])

    return x1, x2

# This function initializes and executes a VFAE
def train_VFAE(train_X, train_Y, train_s, test_X, test_Y, test_s, weights, bias, params, dims):
    # Initialize placeholders
    x = tf.placeholder(tf.float32, shape=[None, dims['x']], name='x')
    s = tf.placeholder(tf.float32, shape=[None, dims['s']], name='s')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    is_sensitive = tf.placeholder(tf.bool, shape=[None, dims['z1']], name='is_sensitive')

    # First encoder
    epsilon0 = tf.random_normal([params['enc1']['out_dim']], dtype=tf.float32, name='epsilon0')
    z1_enc, z1_enc_mu, z1_enc_log_sigma = Gaussian_MLP(tf.concat([x, s], axis=1), weights['enc1'], bias['enc1'],
                                                       params['enc1']['act'], epsilon0)
    # Second encoder
    epsilon1 = tf.random_normal([params['enc2']['out_dim']], dtype=tf.float32, name='epsilon1')
    z2_enc, z2_enc_mu, z2_enc_log_sigma = Gaussian_MLP(tf.concat([z1_enc, y], axis=1), weights['enc2'], bias['enc2'],
                                                       params['enc2']['act'], epsilon1)
    # First decoder
    epsilon2 = tf.random_normal([params['dec1']['out_dim']], dtype=tf.float32, name='epsilon2')
    z1_dec, z1_dec_mu, z1_dec_log_sigma = Gaussian_MLP(tf.concat([z2_enc, y], axis=1), weights['dec1'], bias['dec1'],
                                                       params['dec1']['act'], epsilon2)
    # Second decoder
    epsilon3 = tf.zeros([params['dec2']['out_dim']], dtype=tf.float32, name='epsilon3')
    x_out = \
        Gaussian_MLP(tf.concat([z1_dec, s], axis=1), weights['dec2'], bias['dec2'], params['dec2']['act'], epsilon3)[0]
    # Predictive posterior
    epsilon4 = tf.zeros([params['us']['out_dim']], dtype=tf.float32, name='epsilon4')
    y_us = Gaussian_MLP(z1_enc, weights['us'], bias['us'], params['us']['act'], epsilon4)[0]

    # Calculate the loss function
    KL_z1 = KL(z1_enc_mu, z1_enc_log_sigma, z1_dec_mu, z1_dec_log_sigma)
    KL_z2 = KL(z2_enc_mu, z2_enc_log_sigma)
    # Bernoulli error measure
    LH_x = tf.reduce_sum(
        tf.concat([x, s], axis=1) * tf.log(1e-10 + x_out) + (1 - tf.concat([x, s], axis=1)) * tf.log(1e-10 + 1 - x_out),
        axis=1)

    index = tf.range(tf.shape(y)[0])

    idx = tf.stack([index[:, tf.newaxis], tf.cast(y, tf.int32)], axis=-1)

    LH_y = tf.reduce_sum(tf.log(1e-10 + tf.gather_nd(y_us, idx)), axis=1)

    # Maximum Mean Discrepancy (MMD)
    # Filter z1_env into the parts generated by data for which s=1 and s=0
    z1_enc_s1, z1_enc_s0 = apply_bool_mask(z1_enc, is_sensitive)
    # Apply fast MMD
    MMD = fast_MMD(z1_enc_s1, z1_enc_s0, params)

    loss = -(
        -tf.reduce_mean(KL_z1) - tf.reduce_mean(KL_z2) + tf.reduce_mean(LH_x) - params['alpha'] * tf.reduce_mean(LH_y) -
        params['beta'] * MMD)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

    train = optimizer.minimize(loss)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    index_shuffled = np.arange(train_X.shape[0])
    np.random.shuffle(index_shuffled)

    N_batches = int(float(train_X.shape[0]) / float(params['batch_size']))

    for i in range(params['N_epochs']):

        for b in range(N_batches):
            batch_X, batch_Y, batch_s = get_batch([train_X, train_Y, train_s], params['batch_size'], index_shuffled, b)

            batch_dict = {x: batch_X, y: batch_Y, s: batch_s, is_sensitive: np.tile(batch_s == 1, dims['z1'])}
            full_dict = {x: train_X, y: train_Y, s: train_s, is_sensitive: np.tile(train_s == 1, dims['z1'])}

            sess.run(train, feed_dict=batch_dict)

        if (i % params['print_frequency'] == 0 or i == params['N_epochs'] - 1):
            print("Epoch %s: batch loss = %s and global loss = %s" % (i,
                                                                      sess.run(loss, feed_dict=batch_dict),
                                                                      sess.run(loss, feed_dict=full_dict)))
            print("KL_z1 = %s, KL_z2 = %s, RL = %s, unsupervised posterior = %s, MMD = %s" \
                  % sess.run((tf.reduce_mean(KL_z1),
                              tf.reduce_mean(KL_z2),
                              tf.reduce_mean(LH_x),
                              tf.reduce_mean(LH_y),
                              params['beta'] * MMD
                              ),
                             feed_dict=batch_dict))

    test_dict = {x: test_X, y: test_Y, s: test_s, is_sensitive: np.tile(test_s == 1, dims['z1'])}
    return sess.run([x_out, z1_enc, loss, tf.reduce_mean(LH_x)], feed_dict=test_dict)

def obtain_X_Y_s(df, target_feature, sensitive_features):
    X = np.asarray(df.drop([target_feature] + sensitive_features, axis=1))
    Y = np.expand_dims(np.asarray(df[target_feature]).astype(int), 1)
    s = np.asarray(df[sensitive_features]).astype(int)

    return X, Y, s

