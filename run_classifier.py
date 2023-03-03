import os, argparse, random, atexit
import tensorflow as tf
import numpy as np
from model.modeling import BertConfig, ClassifierBert, SelectStrategy
from model.optimization import CreateOptimizer


### Hyper-parameters and Settings ###
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=str, default='0123')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--is_training', action='store_true')
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--num_labels', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-5, help="2e-5, 3e-5, 4e-5, 5e-5")
parser.add_argument('--warmup_proportion', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=3, help="2,3,4,5")
parser.add_argument('--batch_size', type=int, default=32, help="16, 32, 48")
parser.add_argument('--config', type=str, default="./datasets/bert_config.json")
parser.add_argument('--data_dir', type=str, default="./datasets/", help="load train/val/test datasets")
parser.add_argument('--save_dir', type=str, default="./saved_weights/bert_cls/", help="load pre-trained model weights and save fine-tuned model weights")
parser.add_argument('--log_dir', type=str, default="./logs/bert_cls/", help="save tensorboard logs")
parser.add_argument('--result_dir', type=str, default="./results/", help="export model results")
arg = parser.parse_args()


### Run ###
def main(train_dataset, eval_dataset):
    ### model construction ###
    with strategy.scope():
        model = ClassifierBert(
            is_training=arg.is_training,
            seq_len=arg.seq_len,
            num_labels=arg.num_labels,
            config=arg.config,
            bert_module_weights_path=str("/".join(arg.save_dir.split("/")[:-2]) + 
                                         "/bert_module.pickle") if arg.is_training else None,
        )
        model.summary()
        optimizer = CreateOptimizer(
            warmup_steps=int(arg.warmup_proportion * arg.total_train_steps * arg.epochs), 
            num_train_steps=int(arg.total_train_steps * arg.epochs), 
            initial_learning_rate=arg.lr, 
            weight_decay_rate=0.01, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-6, 
        )
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
    ### model loss and metrics (acc, auc, f1score, ...) ###
    train_loss = tf.keras.metrics.Mean(name="loss")
    infer_loss = tf.keras.metrics.Mean(name="val_loss")
    train_metric = tf.keras.metrics.Accuracy(name="acc")
    infer_metric = tf.keras.metrics.Accuracy(name="val_acc")
    train_summary_writer = tf.summary.create_file_writer(arg.log_dir + "train")
    infer_summary_writer = tf.summary.create_file_writer(arg.log_dir + "val")
    

    ### classifier loss ###
    def get_cls_output_metric_fn(logits, label_ids):
        """ You may consider applying additional processes here if the labels are imbalanced """
        ### multi-class classification ###
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        label_ids = tf.cast(label_ids, dtype=tf.int32)
        one_hot_labels = tf.one_hot(label_ids, depth=arg.num_labels, dtype=tf.float32)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

        ### categorical cross entropy (cce) ###
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predictions, probabilities)


    ### train procedure ###
    def train_step(dataset):
        with tf.GradientTape() as tape:
            input_ids, input_pos, label_ids = dataset
            logits = model([input_ids, input_pos], training=True)
            (loss, predictions, _) = get_cls_output_metric_fn(
                logits, label_ids
            )
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, predictions, label_ids

    @tf.function
    def distributed_train_step(dataset_inputs):
        loss, preds, labels = strategy.run(train_step, args=(dataset_inputs,))
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        preds = strategy.gather(preds, axis=0)
        labels = strategy.gather(labels, axis=0)
        # update loss and metric
        train_loss.update_state(loss)
        train_metric.update_state(
            y_true=labels, 
            y_pred=preds, 
        )


    ### evaluation procedure ###
    def infer_step(dataset):
        input_ids, input_pos, label_ids = dataset
        logits = model([input_ids, input_pos], training=False)
        (loss, predictions, probabilities) = get_cls_output_metric_fn(
            logits, label_ids
        )
        return loss, predictions, probabilities, label_ids

    @tf.function
    def distributed_infer_step(dataset_inputs):
        loss, preds, probs, labels = strategy.run(infer_step, args=(dataset_inputs,))
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        preds = strategy.gather(preds, axis=0)
        probs = strategy.gather(probs, axis=0)
        labels = strategy.gather(labels, axis=0)
        # update loss and metric
        infer_loss.update_state(loss)
        infer_metric.update_state(
            y_true=labels, 
            y_pred=preds, 
        )
        if not arg.is_training: return preds, probs, labels


    ### training loop ###
    if arg.is_training:
        print("Training Start")
        best_loss = 1e+9
        for epoch in range(arg.epochs):
            for step_x, x in enumerate(train_dataset):
                distributed_train_step(x)
                # show training loss & metric logs per 1 step
                print(f'Steps : {step_x+1}/{arg.total_train_steps} Train_Loss : {float(train_loss.result()):.7f} Train_Metric : {float(train_metric.result()):.7f}'
                      , end='\r')
            with train_summary_writer.as_default():
                tf.summary.scalar('Loss', train_loss.result(), step=epoch)
                tf.summary.scalar('Accuracy', train_metric.result(), step=epoch)

            for step_v, v in enumerate(eval_dataset):
                distributed_infer_step(v)
                # show evaluate loss & metric logs per 1 step
                print(f'Steps : {step_v+1}/{arg.total_infer_steps} Valid_Loss : {float(infer_loss.result()):.7f} Valid_Metric : {float(infer_metric.result()):.7f}'
                      , end='\r')
            with infer_summary_writer.as_default():
                tf.summary.scalar('Loss', infer_loss.result(), step=epoch)
                tf.summary.scalar('Accuracy', infer_metric.result(), step=epoch)

            # show loss & metric logs per 1 epoch
            print(f'Epoch : {epoch+1}/{int(arg.epochs)}  '\
                f'Train_Loss : [{float(train_loss.result()):.7f}] Train_Acc : [{float(train_metric.result()):.7f}]  '\
                f'Valid_Loss : [{float(infer_loss.result()):.7f}] Valid_Acc : [{float(infer_metric.result()):.7f}]')

            # save model based on evaluation loss
            if best_loss > infer_loss.result():
                best_loss = infer_loss.result()
                best_model = model
                # save total cls model
                tf.saved_model.save(best_model, arg.save_dir)
                print('Model Saved!')

            # loss/metric reset
            train_loss.reset_states()
            train_metric.reset_states()
            infer_loss.reset_states()
            infer_metric.reset_states()

    else:
        print("Inference Start")
        model = tf.saved_model.load(arg.save_dir)
        print("Saved Model Loaded")
        the_origins = []
        the_preds = []
        the_probs = []
        for step_v, v in enumerate(eval_dataset):
            pd, pb, ll = distributed_infer_step(v)
            the_origins.append(ll)
            the_preds.append(pd)
            the_probs.append(pb)
            # show evaluate loss & metric logs per 1 step
            print(f'Steps : {step_v+1}/{arg.total_infer_steps} Test_Loss : {float(infer_loss.result()):.7f} Test_Metric : {float(infer_metric.result()):.7f}'
                  , end='\r')
        # show loss & metric logs per 1 epoch
        print("")
        print(f'Test_Loss : [{float(infer_loss.result()):.7f}] Test_Acc : [{float(infer_metric.result()):.7f}]')

        # export model inference results into a csv file
        the_origins = np.expand_dims(np.concatenate(the_origins, axis=0), axis=-1)
        the_preds = np.expand_dims(np.concatenate(the_preds, axis=0), axis=-1)
        the_probs = np.concatenate(the_probs, axis=0)
        the_results = np.concatenate((the_origins, the_preds, the_probs), axis=1)
        the_columns = ','.join(['Label_ID','Label_Pred']+['Prob_'+str(n) for n in range(arg.num_labels)])
        np.savetxt(arg.result_dir+"bert_cls_sample.csv", the_results, delimiter=',', header=the_columns, comments='')
        print("Classification Results Exported to:", arg.result_dir)

    print("Done")


if __name__ == "__main__":
    ### Settings ###
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(arg.gpu_num))
    print(tf.config.experimental.list_physical_devices('GPU'))
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Tensorflow version:', tf.__version__)
    tf.keras.backend.clear_session()
    random.seed(arg.seed)
    tf.random.set_seed(arg.seed)
    np.random.seed(arg.seed)
    AUTOTUNE = tf.data.AUTOTUNE
    OPTIONS = tf.data.Options()
    OPTIONS.experimental_deterministic = False
    OPTIONS.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO


    ### FP16, Synchronous data parallelism ###
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    strategy = SelectStrategy()
    arg.batch_size = arg.batch_size * strategy.num_replicas_in_sync


    ### Datasets Loader ###
    arg.config = BertConfig.from_json_file(json_file=arg.config)
    """--------------------------------------"""
    """----- insert input datasets here -----"""
    """--------------------------------------"""


    ### Prepare Datasets ###
    if arg.is_training:
        train_dataset = (train_dataset.shuffle(buffer_size=len(train_dataset)).batch(arg.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        eval_dataset = (eval_dataset.batch(arg.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
        eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)
    else:
        train_dataset = None
        arg.total_train_steps = 0
        eval_dataset = (eval_dataset.batch(arg.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
        eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)

        
    ### Run ###
    main(train_dataset=train_dataset, eval_dataset=eval_dataset)
    if strategy.num_replicas_in_sync > 1: atexit.register(strategy._extended._collective_ops._pool.close)
