import os
from sys import path
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
import random
import itertools

from DFMnet import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# Training parameterss
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate.')
flags.DEFINE_integer('batch_size', 4, 'batch size.')

# Architecture parameters
flags.DEFINE_integer('num_layers', 7, 'network depth')
flags.DEFINE_integer('num_evecs', 120, "number of eigenvectors used for representation")
flags.DEFINE_integer('dim_shot', 352, '')
flags.DEFINE_integer('num_ch', 3, '')

# Data parameters
flags.DEFINE_multi_string('targets_dirs', [
                    'Shapes/alpha00',
                    'Shapes/alpha06',
                    'Shapes/alpha08', ],
                    'directory with shapes')
flags.DEFINE_string('files_name', 'tr_reg_', 'name common to all the shapes')
flags.DEFINE_integer('max_train_iter', 10000, '')
flags.DEFINE_integer('num_vertices', 3000, '')
flags.DEFINE_integer('save_summaries_secs', 500, '')
flags.DEFINE_integer('save_model_secs', 500, '')

# FAUST:
if FLAGS.files_name == 'tr_reg_':
    train_subjects = list(range(0, 80))
    train_subjects_pairs = list(itertools.permutations(train_subjects, 2))
else:
    raise Exception(f'Unexpected files name - {FLAGS.files_name}')

n_tr = len(train_subjects)

def get_input_pair(batch_size, num_vertices):
    batch_input = {
        'source_evecs': np.zeros((batch_size, FLAGS.num_ch, num_vertices, FLAGS.num_evecs)),
        'target_evecs': np.zeros((batch_size, FLAGS.num_ch, num_vertices, FLAGS.num_evecs)),
        'source_evecs_trans': np.zeros((batch_size, FLAGS.num_ch, FLAGS.num_evecs, num_vertices)),
        'target_evecs_trans': np.zeros((batch_size, FLAGS.num_ch, FLAGS.num_evecs, num_vertices)),
        'source_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),
        'target_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),
        'source_evals': np.zeros((batch_size, FLAGS.num_ch, FLAGS.num_evecs)),
        'target_evals': np.zeros((batch_size, FLAGS.num_ch, FLAGS.num_evecs))
    }

    for i_batch in range(batch_size):
        i_source, i_target = random.choice(train_subjects_pairs)

        batch_input_ = get_pair_from_ram(i_target, i_source)

        batch_input_['source_labels'] = range(np.shape(batch_input_['source_evecs'])[1])
        batch_input_['target_labels'] = range(np.shape(batch_input_['target_evecs'])[1])
        joint_lbls = np.intersect1d(batch_input_['source_labels'],
                                    batch_input_['target_labels'])
        joint_labels_source = np.random.permutation(joint_lbls)[:num_vertices]
        joint_labels_target = np.random.permutation(joint_lbls)[:num_vertices]

        ind_dict_source = {value: ind for ind, value in enumerate(batch_input_['source_labels'])}
        ind_source = [ind_dict_source[x] for x in joint_labels_source]

        ind_dict_target = {value: ind for ind, value in enumerate(batch_input_['target_labels'])}
        ind_target = [ind_dict_target[x] for x in joint_labels_target]
        message = "number of indices must be equal"
        assert len(ind_source) == len(ind_target), message

        evecs = batch_input_['source_evecs'][:, ind_source, :]
        evecs_trans = batch_input_['source_evecs_trans'][:, :, ind_source]
        shot = batch_input_['source_shot'][ind_source, :]
        evals = [item for sublist in batch_input_['source_evals'] for item in sublist]
        batch_input['source_evecs'][i_batch] = evecs
        batch_input['source_evecs_trans'][i_batch] = evecs_trans
        batch_input['source_shot'][i_batch] = shot
        batch_input['source_evals'][i_batch] = evals

        evecs = batch_input_['target_evecs'][:, ind_target, :]
        evecs_trans = batch_input_['target_evecs_trans'][:, :, ind_target]
        shot = batch_input_['target_shot'][ind_target, :]
        evals = [item for sublist in batch_input_['target_evals'] for item in sublist]
        batch_input['target_evecs'][i_batch] = evecs
        batch_input['target_evecs_trans'][i_batch] = evecs_trans
        batch_input['target_shot'][i_batch] = shot
        batch_input['target_evals'][i_batch] = evals

    return batch_input


def get_pair_from_ram(i_target, i_source):
    input_data = {
        'source_evecs': targets_train[i_source]['target_evecs'],
        'source_evecs_trans': targets_train[i_source]['target_evecs_trans'],
        'source_shot': targets_train[i_source]['target_shot'],
        'source_evals': targets_train[i_source]['target_evals'],
    }
    input_data.update(targets_train[i_target])

    return input_data


targets_train = {}
def load_targets_to_ram():

    for i in train_subjects:
        mats = [sio.loadmat(f'{targets_dir}/{FLAGS.files_name}{i:03}.mat')
                for targets_dir in FLAGS.targets_dirs]
        targets_train[i] = {
            'target_evecs': np.stack([m['target_evecs'][:, 0:FLAGS.num_evecs] for m in mats]),
            'target_evecs_trans': np.stack([m['target_evecs_trans'][0:FLAGS.num_evecs, :] for m in mats]),
            'target_evals': np.stack([m['target_evals'][:, 0:FLAGS.num_evecs] for m in mats]),
            'target_shot': mats[0]['target_shot']
        }

def run_training():

    print('log_dir=%s' % FLAGS.log_dir)
    if not os.path.isdir(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    print('num_evecs=%d' % FLAGS.num_evecs)

    print('building graph...')
    with tf.Graph().as_default():

        # Set placeholders for inputs
        source_shot = tf.placeholder(tf.float32,
                                     shape=(None, None, FLAGS.dim_shot),
                                     name='source_shot')
        target_shot = tf.placeholder(tf.float32,
                                     shape=(None, None, FLAGS.dim_shot),
                                     name='target_shot')
        source_evecs = tf.placeholder(tf.float32,
                                      shape=(None, FLAGS.num_ch, None, FLAGS.num_evecs),
                                      name='source_evecs')
        source_evecs_trans = tf.placeholder(
                                      tf.float32,
                                      shape=(None, FLAGS.num_ch, FLAGS.num_evecs, None),
                                      name='source_evecs_trans')
        source_evals = tf.placeholder(tf.float32,
                                      shape=(None, FLAGS.num_ch, FLAGS.num_evecs),
                                      name='source_evals')
        target_evecs = tf.placeholder(tf.float32,
                                      shape=(None, FLAGS.num_ch, None, FLAGS.num_evecs),
                                      name='target_evecs')
        target_evecs_trans = tf.placeholder(
                                      tf.float32,
                                      shape=(None, FLAGS.num_ch, FLAGS.num_evecs, None),
                                      name='target_evecs_trans')
        target_evals = tf.placeholder(tf.float32,
                                      shape=(None, FLAGS.num_ch, FLAGS.num_evecs),
                                      name='target_evals')

        # train\test switch flag
        phase = tf.placeholder(dtype=tf.bool, name='phase')

        net_loss, safeguard_inverse, merged, net = dfmnet_model(
                                            phase, source_shot, target_shot,
                                            source_evecs, source_evecs_trans,
                                            source_evals, target_evecs,
                                            target_evecs_trans, target_evals
                                                                )

        summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_op = optimizer.minimize(net_loss,
                                      global_step=global_step,
                                      aggregation_method=2)

        saver = tf.train.Saver(max_to_keep=40)
        sv = tf.train.Supervisor(
                                logdir=FLAGS.log_dir,
                                init_op=tf.global_variables_initializer(),
                                local_init_op=tf.local_variables_initializer(),
                                global_step=global_step,
                                save_summaries_secs=FLAGS.save_summaries_secs,
                                save_model_secs=FLAGS.save_model_secs,
                                summary_op=None,
                                saver=saver
                                 )

        writer = sv.summary_writer

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print('starting session...')

        iteration = 0
        with sv.managed_session(config=config) as sess:
            print('loading data to ram...')
            load_targets_to_ram()
            print('starting training loop...')
            while not sv.should_stop() and iteration < FLAGS.max_train_iter:
                iteration += 1
                start_time = time.time()
                input_data = get_input_pair(FLAGS.batch_size,
                                            FLAGS.num_vertices)
                feed_dict = {
                    phase: True,
                    source_shot: input_data['source_shot'],
                    target_shot: input_data['target_shot'],
                    source_evecs: input_data['source_evecs'],
                    source_evecs_trans: input_data['source_evecs_trans'],
                    source_evals: input_data['source_evals'],
                    target_evecs: input_data['target_evecs'],
                    target_evecs_trans: input_data['target_evecs_trans'],
                    target_evals: input_data['target_evals']
                             }

                run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
                summaries, step, my_loss, safeguard, _ = sess.run([merged, global_step, net_loss, safeguard_inverse, train_op],
                                                                  feed_dict=feed_dict, options=run_options)
                writer.add_summary(summaries, step)
                summary_ = sess.run(summary)
                writer.add_summary(summary_, step)

                duration = time.time() - start_time
                print('train - step %d: loss = %.2f (%.3f sec)'
                      % (step, my_loss, duration))

            saver.save(sess, FLAGS.log_dir + '/model.ckpt', global_step=step)
            writer.flush()
            sv.request_stop()
            sv.stop()


def main(_):
    import time
    start_time = time.time()
    run_training()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()

