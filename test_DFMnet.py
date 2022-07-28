import os
import time
import tensorflow as tf
import scipy.io as sio
import numpy as np
import itertools
from scipy.spatial import cKDTree

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_evecs', 120, 'number of eigenvectors used for representation')
flags.DEFINE_integer('num_model', 10000, '')
flags.DEFINE_integer('num_ch', 3, '')
flags.DEFINE_multi_string('targets_dirs', [
                    'Shapes/alpha00',
                    'Shapes/alpha06',
                    'Shapes/alpha08', ],
                    'directory with shapes')
flags.DEFINE_string('files_name', 'tr_reg_', 'name common to all the shapes')
flags.DEFINE_string('matches_dir', './Matches/SCAPE_r/1500/', 'directory to matches')

def get_test_pair_source(i):
    mats = [sio.loadmat(f'{targets_dir}/{FLAGS.files_name}{i:03}.mat')
            for targets_dir in FLAGS.targets_dirs]
    input_data = {
        'source_evecs': np.stack([m['target_evecs'][:, 0:FLAGS.num_evecs] for m in mats]),
        'source_evecs_trans': np.stack([m['target_evecs_trans'][0:FLAGS.num_evecs, :] for m in mats]),
        'source_evals': np.stack([m['target_evals'][:, 0:FLAGS.num_evecs] for m in mats]),
        'source_shot': mats[0]['target_shot']
    }
    input_data['source_evals'] = np.transpose(input_data['source_evals'], [1, 0, 2])
    return input_data


def get_test_pair_target(i):
    mats = [sio.loadmat(f'{targets_dir}/{FLAGS.files_name}{i:03}.mat')
            for targets_dir in FLAGS.targets_dirs]
    input_data = {
        'target_evecs': np.stack([m['target_evecs'][:, 0:FLAGS.num_evecs] for m in mats]),
        'target_evecs_trans': np.stack([m['target_evecs_trans'][0:FLAGS.num_evecs, :] for m in mats]),
        'target_evals': np.stack([m['target_evals'][:, 0:FLAGS.num_evecs] for m in mats]),
        'target_shot': mats[0]['target_shot']
    }
    input_data['target_evals'] = np.transpose(input_data['target_evals'], [1, 0, 2])
    return input_data


def run_test():
    os.makedirs(FLAGS.matches_dir, exist_ok=True)
    # Start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print('restoring graph...')
    saver = tf.train.import_meta_graph('%smodel.ckpt-%s.meta' % (FLAGS.log_dir, FLAGS.num_model))
    saver.restore(sess, tf.train.latest_checkpoint('%s' % FLAGS.log_dir))
    graph = tf.get_default_graph()
    # Retrieve placeholder variables
    source_evecs = graph.get_tensor_by_name('source_evecs:0')
    source_evecs_trans = graph.get_tensor_by_name('source_evecs_trans:0')
    target_evecs = graph.get_tensor_by_name('target_evecs:0')
    target_evecs_trans = graph.get_tensor_by_name('target_evecs_trans:0')
    source_shot = graph.get_tensor_by_name('source_shot:0')
    target_shot = graph.get_tensor_by_name('target_shot:0')
    phase = graph.get_tensor_by_name('phase:0')
    source_evals = graph.get_tensor_by_name('source_evals:0')
    target_evals = graph.get_tensor_by_name('target_evals:0')

    Ct_est = graph.get_tensor_by_name(
                    'matrix_solve_ls/cholesky_solve/MatrixTriangularSolve_1:0'
                                      )

    pairs = []

    # FAUST:
    if FLAGS.files_name == 'tr_reg_':
        pairs += list(itertools.combinations(range(80, 90), 2)) + \
                list(itertools.combinations(range(90, 100), 2)) + \
                list(itertools.product(range(80, 90), range(90, 100)))
    else:
        raise Exception(f'Unexpected files name - {FLAGS.files_name}')

    for i, j in pairs:
        t = time.time()
        try:
            input_data_source = get_test_pair_source(i)
            source_evecs_ = input_data_source['source_evecs'][:, 0:FLAGS.num_evecs]
            input_data_target = get_test_pair_target(j)
        except FileNotFoundError as fe:
                print(f'File not found: {fe.filename}, skipping')
                continue


        feed_dict = {
            phase: True,
            source_shot: [input_data_source['source_shot']],
            target_shot: [input_data_target['target_shot']],
            source_evecs: [input_data_source['source_evecs'][:,:,0:FLAGS.num_evecs]],
            source_evecs_trans: [input_data_source['source_evecs_trans'][:,0:FLAGS.num_evecs,:]],
            source_evals: input_data_source['source_evals'][:,0:FLAGS.num_evecs],
            target_evecs: [input_data_target['target_evecs'][:,:, 0:FLAGS.num_evecs]],
            target_evecs_trans: [input_data_target['target_evecs_trans'][:, 0:FLAGS.num_evecs,:]],
            target_evals: input_data_target['target_evals'][:, 0: FLAGS.num_evecs],
        }

        Ct_est_ = sess.run([Ct_est], feed_dict=feed_dict)
        Ct = np.squeeze(Ct_est_) #Keep transposed

        np.save(FLAGS.matches_dir +
                FLAGS.files_name + '%.3d-' % i +
                FLAGS.files_name + '%.3d' % j,
                Ct)

        kdtrees = [cKDTree(input_data_source['source_evecs'][ch] @ Ct[ch]) for ch in range(FLAGS.num_ch)]
        dists = [None] * FLAGS.num_ch
        indices = [None] * FLAGS.num_ch
        for ch in range(FLAGS.num_ch):
            dists[ch], indices[ch] = kdtrees[ch].query(input_data_target['target_evecs'][ch], n_jobs=-1)
            # normalize distances to [0,1]
            dists[ch] /= dists[ch].max()
            dists[ch] -= dists[ch].mean()

        n = input_data_source['source_evecs'][0].shape[0]
        best_indices = -np.ones(n, dtype=np.int)
        best_dists = 2*np.ones(n)

        for ch in range(FLAGS.num_ch):
            loc = dists[ch] < best_dists
            best_dists[loc] = dists[ch][loc]
            best_indices[loc] = indices[ch][loc]

        best_indices = best_indices + 1

        print("Computed correspondences for pair: %s, %s." % (i, j) +" Took %f seconds" % (time.time() - t))

        params_to_save = {}
        params_to_save['matches'] = best_indices
        #params_to_save['C'] = Ct.T
        # For Matlab where index start at 1
        sio.savemat(FLAGS.matches_dir +
                    FLAGS.files_name + '%.3d-' % i +
                    FLAGS.files_name + '%.3d.mat' % j, params_to_save)


def main(_):
    import time
    start_time = time.time()
    run_test()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    tf.app.run()

