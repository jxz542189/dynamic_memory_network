import tensorflow as tf
from model.dmn.graph import Graph
import multiprocessing
import json
import shutil
import os
from model.dmn.model import Model
from utils.data_loader import DataLoader
import numpy as np


tf.reset_default_graph()
from hbconfig import Config

config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
params_path = os.path.join(config_path, 'bAbi_task1.yml')
Config(params_path)
os.environ['CUDA_VISIBLE_DEVICES'] = Config.train.CUDA_VISIBLE_DEVICES
print("Config: ", Config)
# max_facts_seq_len = None


RESUME_TRAINING  = False
model_dir = 'trained_models/{}'.format(Config.train.model_dir)
run_config = tf.estimator.RunConfig(log_step_count_steps=Config.train.log_step_count_steps,
                                    tf_random_seed=Config.train.tf_random_seed,
                                    model_dir=model_dir)
print(run_config)

data_loader = DataLoader(
    task_path=Config.data.task_path,
    task_id=Config.data.task_id,
    task_test_id=Config.data.task_id,
    w2v_dim=Config.model.embed_dim,
    use_pretrained=Config.model.use_pretrained
)
data = data_loader.make_train_and_test_set()
total_steps = int(int(len(data['train'][0]) / Config.model.batch_size) * Config.train.n_epochs)
test_steps = int(len(data['test'][0]) / Config.model.batch_size)
vocab_size = len(data_loader.vocab)
max_facts_seq_len = data_loader.max_facts_seq_len
max_input_mask_len = data_loader.max_input_mask_len
max_question_seq_len = data_loader.max_question_seq_len
Config.data.max_facts_seq_len = data_loader.max_facts_seq_len
Config.data.max_question_seq_len = data_loader.max_question_seq_len
Config.data.max_input_mask_length = data_loader.max_input_mask_len
params = tf.contrib.training.HParams(max_facts_seq_len=max_facts_seq_len,
                                     max_input_mask_len=max_input_mask_len,
                                     max_question_seq_len=max_question_seq_len,
                                     **Config.model.to_dict())
print(params)
EVAL_AFTER_SEC = 60
def input_fn(data,
             mode=tf.estimator.ModeKeys.EVAL,
             num_epochs=1,
             batch_size=200):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count()
    buffer_size = 2 * batch_size + 1
    print("")
    print("* data input_fn:")
    print("================")
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    #train_input:(10000,68,50) train_input_mask:(10000,10), train_question:(10000,3,50) train_answer:(10000, 1)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_input, train_question, train_answer, train_input_mask = data['train']
    else:
        train_input, train_question, train_answer, train_input_mask = data['test']

    # print(data_loader.max_facts_seq_len)
    # print(data_loader.max_input_mask_len)
    # print(data_loader.max_question_seq_len)
    # res = np.concatenate([train_input, train_input_mask, train_question], axis=-1)
    # res = [train_input, train_input_mask, train_question]
    dataset = tf.data.Dataset.from_tensor_slices(({"input_data":train_input, "input_data_mask":train_input_mask,"question_data":train_question}, train_answer))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat(None)
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size)

    return dataset


def model_fn(features, labels, mode, params):
    # max_facts_seq_len = params.max_facts_seq_len
    # max_input_mask_len = params.max_input_mask_len
    # max_question_seq_len = params.max_question_seq_len
    train_input, train_question, train_input_mask = features["input_data"], features["question_data"], features["input_data_mask"]
    train_answer = labels
    graph = Graph(mode, max_input_mask_len, vocab_size)
    output = graph.build(embedding_input=train_input,
                         input_mask=train_input_mask,
                         embedding_question=train_question)
    predictions = tf.argmax(output, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {
            'predictions': predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)
    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            train_answer,
            output,
            scope="cross-entropy")
        reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        loss = tf.add(cross_entropy, reg_term)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss,
                                                   tf.train.get_global_step(),
                                                   optimizer=Config.train.get("optimizer", "Adam"),
                                                   learning_rate=Config.train.learning_rate,
                                                   summaries=['loss', 'gradients', 'learning_rate'],
                                                   name="train_op"
                                                   )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = {
            'accuracy': tf.metrics.accuracy(train_answer, predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)


def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=hparams,
                                       config=run_config)
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator


# def serving_input_fn():
#     receiver_tensor = {
#         'instances': tf.placeholder(tf.int32, [None, None])
#     }
#     features = {
#         key: tensor
#         for key, tensor in receiver_tensor.items()
#     }
#
#     return tf.estimator.export.ServingInputReceiver(
#         features, receiver_tensor)



if __name__ == '__main__':
    # ==============训练方式===============
    if not RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        print("Resuming training...")
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)

    # run_config = tf.estimator.RunConfig(log_step_count_steps=config.train['log_step_count_steps'],
    #                                     tf_random_seed=config.train['tf_random_seed'],
    #                                     model_dir=model_dir,
    #                                     )

    run_config = tf.estimator.RunConfig(log_step_count_steps=Config.train.log_step_count_steps,
                                        tf_random_seed=Config.train.tf_random_seed,
                                        model_dir=model_dir,
                                        session_config=tf.ConfigProto(allow_soft_placement=True,
                                                                      log_device_placement=True),
                                        train_distribute=distribution)
    estimator = create_estimator(run_config, params)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(data,
                                  mode=tf.estimator.ModeKeys.TRAIN,
                                  num_epochs=Config.train.n_epochs,
                                  batch_size=Config.model.batch_size),
        max_steps=total_steps,
        hooks=None
    )
    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.EVAL,
    #                               batch_size=Config.model.batch_size),
    #     exporters=[tf.estimator.LatestExporter(name="predict",
    #                                            serving_input_receiver_fn=serving_input_fn,
    #                                            exports_to_keep=1,
    #                                            as_text=True)],
    #     steps=test_steps,
    #     throttle_secs=EVAL_AFTER_SEC
    # )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(data,
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_size=Config.model.batch_size),
        steps=None,
        throttle_secs=EVAL_AFTER_SEC
    )
    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)




