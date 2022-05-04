import argparse
import pickle
import sys

import numpy as np

from data_prep import batch_iter, createOneHotMosei2way, get_raw_data, createOneHot

seed = 1234

np.random.seed(seed)
import tensorflow as tf
from tqdm import tqdm

from model import LSTM_Model

from sklearn.metrics import f1_score

tf.set_random_seed(seed)
# tf.random.set_seed(seed)

unimodal_activations = {}


def multimodal():
    # if use_raw:
    #     if attn_fusion:
    #         attn_fusion = False

    train_data, test_data, train_cs, test_cs, train_english, test_english, train_label, test_label, seqlen_train, seqlen_test = get_raw_data()

    # else:
    #     print("starting multimodal")
    #     # Fusion (appending) of features

    #     train_cs = unimodal_activations['train_cs']
    #     train_english = unimodal_activations['train_english']
        
    #     test_cs = unimodal_activations['test_cs']
    #     test_english = unimodal_activations['test_english']
        
    #     train_mask = unimodal_activations['train_mask']
    #     test_mask = unimodal_activations['test_mask']

    #     print('train_mask', train_mask.shape)

    #     train_label = unimodal_activations['train_label']
    #     print('train_label', train_label.shape)
    #     test_label = unimodal_activations['test_label']
    #     print('test_label', test_label.shape)

    #     # print(train_mask_bool)
    #     seqlen_train = np.sum(train_mask, axis=-1)
    #     print('seqlen_train', seqlen_train.shape)
    #     seqlen_test = np.sum(test_mask, axis=-1)
    #     print('seqlen_test', seqlen_test.shape)

    cs_dim = train_cs.shape[-1]
    english_dim = train_english.shape[-1]
    # if attn_fusion:
    #     print('With attention fusion')
    allow_soft_placement = True
    log_device_placement = False

    # Multimodal model
    session_conf = tf.ConfigProto(
        # device_count={'GPU': gpu_count},
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_device = 0
    best_acc = 0
    best_loss_accuracy = 0
    best_loss = 10000000.0
    best_epoch = 0
    best_epoch_loss = 0
    with tf.device('/device:GPU:%d' % gpu_device):
        print('Using GPU - ', '/device:GPU:%d' % gpu_device)
        with tf.Graph().as_default():
            tf.set_random_seed(seed)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # print('train_english.shape: ', train_english.shape)
                model = LSTM_Model(train_english.shape[1:], 0.0001, a_dim=cs_dim, t_dim=english_dim,
                                   emotions=2, attn_fusion=True,
                                   unimodal=False, enable_attn_2=True,
                                   seed=seed)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                print('test_cs.shape: ', test_cs.shape)
                print('test_english.shape: ', test_english.shape)
                print('test_label shape: ', test_label.shape)
                print('test_label: ', test_label)
                # test_cs = np.pad(test_cs, [(0, 512 - test_cs.shape[0]), (0, 512 - test_cs.shape[1])], mode='constant')
                # test_cs = np.reshape(test_cs, (-1, 512, 512))
                # test_english = np.pad(test_english, [(0, 512 - test_english.shape[0]), (0, 512 - test_english.shape[1])], mode='constant')
                # test_english = np.reshape(test_english, (-1, 512, 512))

                test_feed_dict = {
                    model.a_input: test_cs,
                    model.t_input: test_english,
                    model.y: test_label,
                    model.seq_len: seqlen_test,
                    model.lstm_dropout: 0.0,
                    model.lstm_inp_dropout: 0.0,
                    model.dropout: 0.0,
                    model.dropout_lstm_out: 0.0
                }

                # print('\n\nDataset: %s' % (data))
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

                for epoch in range(epochs):
                    epoch += 1

                    batches = batch_iter(list(
                        zip(train_cs, train_english, seqlen_train, train_label)),
                        batch_size)

                    # Training loop. For each batch...
                    print('\nTraining epoch {}'.format(epoch))
                    l = []
                    a = []
                    for i, batch in tqdm(enumerate(batches)):
                        b_train_cs, b_train_english, b_seqlen_train, b_train_label = zip(
                            *batch)
                        # print('batch_hist_v', len(batch_utt_v))
                        feed_dict = {
                            model.a_input: b_train_cs,
                            model.t_input: b_train_english,
                            model.y: b_train_label,
                            model.seq_len: b_seqlen_train,
                            model.lstm_dropout: 0.4,
                            model.lstm_inp_dropout: 0.0,
                            model.dropout: 0.2,
                            model.dropout_lstm_out: 0.2
                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, accuracy {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    step, loss, accuracy, preds, y= sess.run(
                        [model.global_step, model.loss, model.accuracy, model.preds, model.y],
                        test_feed_dict)
                    # f1 = f1_score(np.ndarray.flatten(tf.argmax(y, -1, output_type=tf.int32).eval()),
                    #               np.ndarray.flatten(tf.argmax(preds, -1, output_type=tf.int32).eval()),
                    #               sample_weight=np.ndarray.flatten(tf.cast(mask, tf.int32).eval()), average="weighted")
                    
                    f1 = f1_score(np.ndarray.flatten(tf.argmax(y, -1, output_type=tf.int32).eval()),
                                  np.ndarray.flatten(tf.argmax(preds, -1, output_type=tf.int32).eval()), average="binary")
                    
                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(epoch, step,
                                                                                               loss / test_label.shape[
                                                                                                   0],
                                                                                               accuracy, f1))
                    if accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = accuracy
                    if loss < best_loss:
                        best_loss = loss
                        best_loss_accuracy = accuracy
                        best_epoch_loss = epoch

                print(
                    "\n\nBest epoch: {}\nBest test accuracy: {}\nBest epoch loss: {}\nBest test accuracy when loss is least: {}".format(
                        best_epoch, best_acc, best_epoch_loss, best_loss_accuracy))


def unimodal():
    print(('starting unimodal ', mode))

    # with open('./mosei/text_glove_average.pickle', 'rb') as handle:
    if data == 'mosei' or data == 'mosi':
        with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            # (train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = u.load()
            if data == 'mosei':
                (train_data, train_label, _, _, test_data, test_label, _, train_length, _, test_length, _, _,
                 _) = u.load()
                if classes == '2':
                    train_label, test_label = createOneHotMosei2way(train_label, test_label)
            elif data == 'mosi':
                (train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = u.load()
                train_label = train_label.astype('int')
                test_label = test_label.astype('int')
                train_label, test_label = createOneHot(train_label, test_label)
            else:
                raise NotImplementedError('Unknown dataset...')

            train_label = train_label.astype('int')
            test_label = test_label.astype('int')

            train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
            for i in range(len(train_length)):
                train_mask[i, :train_length[i]] = 1.0

            test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
            for i in range(len(test_length)):
                test_mask[i, :test_length[i]] = 1.0
    elif data == 'iemocap':
        train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_raw_data(
            data, classes)
        if mode == 'text':
            train_data = text_train
            test_data = text_test
        elif mode == 'audio':
            train_data = audio_train
            test_data = audio_test
        elif mode == 'video':
            train_data = video_train
            test_data = video_test

    # train_label, test_label = createOneHotMosei3way(train_label, test_label)

    attn_fusion = False

    print('train_mask', train_mask.shape)

    # print(train_mask_bool)
    seqlen_train = np.sum(train_mask, axis=-1)
    print('seqlen_train', seqlen_train.shape)
    seqlen_test = np.sum(test_mask, axis=-1)
    print('seqlen_test', seqlen_test.shape)

    allow_soft_placement = True
    log_device_placement = False

    # Multimodal model
    session_conf = tf.ConfigProto(
        # device_count={'GPU': gpu_count},
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_device = 0
    best_acc = 0
    best_epoch = 0
    best_loss = 1000000.0
    best_epoch_loss = 0
    is_unimodal = True
    with tf.device('/device:GPU:%d' % gpu_device):
        print('Using GPU - ', '/device:GPU:%d' % gpu_device)
        with tf.Graph().as_default():
            tf.set_random_seed(seed)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = LSTM_Model(train_data.shape[1:], 0.0001, a_dim=0, v_dim=0, t_dim=0, emotions=classes,
                                   attn_fusion=attn_fusion, unimodal=is_unimodal, seed=seed)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                test_feed_dict = {
                    model.input: test_data,
                    model.y: test_label,
                    model.seq_len: seqlen_test,
                    model.mask: test_mask,
                    model.lstm_dropout: 0.0,
                    model.lstm_inp_dropout: 0.0,
                    model.dropout: 0.0,
                    model.dropout_lstm_out: 0.0

                }
                train_feed_dict = {
                    model.input: train_data,
                    model.y: train_label,
                    model.seq_len: seqlen_train,
                    model.mask: train_mask,
                    model.lstm_dropout: 0.0,
                    model.lstm_inp_dropout: 0.0,
                    model.dropout: 0.0,
                    model.dropout_lstm_out: 0.0

                }
                # print('\n\nDataset: %s' % (data))
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

                for epoch in range(epochs):
                    epoch += 1

                    batches = batch_iter(list(
                        zip(train_data, train_mask, seqlen_train, train_label)),
                        batch_size)

                    # Training loop. For each batch...
                    print('\nTraining epoch {}'.format(epoch))
                    l = []
                    a = []
                    for i, batch in tqdm(enumerate(batches)):
                        b_train_data, b_train_mask, b_seqlen_train, b_train_label = zip(
                            *batch)
                        feed_dict = {
                            model.input: b_train_data,
                            model.y: b_train_label,
                            model.seq_len: b_seqlen_train,
                            model.mask: b_train_mask,
                            model.lstm_dropout: 0.4,
                            model.lstm_inp_dropout: 0.0,
                            model.dropout: 0.2,
                            model.dropout_lstm_out: 0.2

                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, accuracy {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    step, loss, accuracy, test_activations = sess.run(
                        [model.global_step, model.loss, model.accuracy, model.inter1],
                        test_feed_dict)
                    loss = loss / test_label.shape[0]
                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}".format(epoch, step, loss, accuracy))

                    if accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = accuracy

                    if epoch == 30:
                        step, loss, accuracy, train_activations = sess.run(
                            [model.global_step, model.loss, model.accuracy, model.inter1],
                            train_feed_dict)
                        unimodal_activations[mode + '_train'] = train_activations
                        unimodal_activations[mode + '_test'] = test_activations

                        unimodal_activations['train_mask'] = train_mask
                        unimodal_activations['test_mask'] = test_mask
                        unimodal_activations['train_label'] = train_label
                        unimodal_activations['test_label'] = test_label

                    if loss < best_loss:
                        best_epoch_loss = epoch
                        best_loss = loss
                        # step, loss, accuracy, train_activations = sess.run(
                        # [model.global_step, model.loss, model.accuracy, model.inter1],
                        # train_feed_dict)
                        # unimodal_activations[mode + '_train'] = train_activations
                        # unimodal_activations[mode + '_test'] = test_activations

                        # unimodal_activations['train_mask'] = train_mask
                        # unimodal_activations['test_mask'] = test_mask
                        # unimodal_activations['train_label'] = train_label
                        # unimodal_activations['test_label'] = test_label

                print("\n\nBest epoch: {}\nBest test accuracy: {}".format(best_epoch, best_acc))
                print("\n\nBest epoch: {}\nBest test loss: {}".format(best_epoch_loss, best_loss))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # argv = sys.argv[1:]
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--unimodal", type=str2bool, nargs='?', const=True, default=True)
    # parser.add_argument("--fusion", type=str2bool, nargs='?', const=True, default=False)
    # parser.add_argument("--attention_2", type=str2bool, nargs='?', const=True, default=False)
    # parser.add_argument("--use_raw", type=str2bool, nargs='?', const=True, default=False)
    # parser.add_argument("--data", type=str, default='mosi')
    # parser.add_argument("--classes", type=str, default='2')
    # args, _ = parser.parse_known_args(argv)

    # print(args)

    batch_size = 20
    # epochs = 100
    # emotions = args.classes
    # assert args.data in ['mosi', 'mosei', 'iemocap']

    # if args.unimodal:
    #     print("Training unimodals first")
    #     modality = ['text', 'audio', 'video']
    #     for mode in modality:
    #         unimodal(mode, args.data, args.classes)

    #     print("Saving unimodal activations")
    #     with open('unimodal_{0}_{1}way.pickle'.format(args.data, args.classes), 'wb') as handle:
    #         pickle.dump(unimodal_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # if not args.use_raw:
    #     with open('unimodal_{0}_{1}way.pickle'.format(args.data, args.classes), 'rb') as handle:
    #         u = pickle._Unpickler(handle)
    #         u.encoding = 'latin1'
    #         unimodal_activations = u.load()

    epochs = 15
    multimodal()
