import os, argparse, random, pickle, atexit
import tensorflow as tf
from model.modeling import BertConfig, MLMBert, SelectStrategy
from model.optimization import CreateOptimizer


### Hyper-parameters and Settings ###
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=str, default='0123')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--is_training', action='store_true')
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--max_pred_per_seq', type=int, default=77, help="set this to around max_seq_length * masked_lm_prob")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--warmup_proportion', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=256, help="batch size for one GPU")
parser.add_argument('--config', type=str, default="./datasets/bert_config.json")
parser.add_argument('--data_dir', type=str, default="./datasets/", help="load train/val/test datasets")
parser.add_argument('--save_dir', type=str, default="./saved_weights/bert_mlm/", help="save pre-trained model weights")
parser.add_argument('--log_dir', type=str, default="./logs/bert_mlm/", help="save tensorboard logs")
arg = parser.parse_args()


### Run ###
def main(train_dataset, eval_dataset):
    ### model construction ###
    with strategy.scope():
        model = MLMBert(
            is_training=arg.is_training,
            batch_size=arg.batch_size,
            seq_len=arg.seq_len,
            max_pred_per_seq=arg.max_pred_per_seq,
            config=arg.config,
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
    ### model loss and metrics ###
    train_loss = tf.keras.metrics.Mean(name="loss")
    train_metric = tf.keras.metrics.Accuracy(name="acc")
    infer_loss = tf.keras.metrics.Mean(name="val_loss")
    infer_metric = tf.keras.metrics.Accuracy(name="val_acc")
    train_summary_writer = tf.summary.create_file_writer(arg.log_dir + "train")
    infer_summary_writer = tf.summary.create_file_writer(arg.log_dir + "val")
    

    ### mlm loss ###
    def get_masked_lm_output_metric_fn(log_probs, masked_lm_ids, masked_lm_weights):
        """
        There are two methods in the original code: get_masked_lm_output & get_next_sentence_output
        Since the model architecture is construced WITHOUT NSP task, mlm loss is the total final loss of the model.
        The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
        """
        ### fix dtypes ###
        log_probs = tf.cast(log_probs, dtype=tf.float32)
        masked_lm_ids = tf.cast(masked_lm_ids, dtype=tf.int32)
        masked_lm_weights = tf.cast(masked_lm_weights, dtype=tf.float32)
        
        ### flatten and one-hot vocab ###
        label_ids = tf.reshape(masked_lm_ids, [-1])
        label_weights = tf.reshape(masked_lm_weights, [-1])
        one_hot_labels = tf.one_hot(label_ids, depth=arg.config.vocab_size, dtype=tf.float32)
        
        ### compute loss ###
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
        label_preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        return (loss, label_preds, label_ids, label_weights)


    ### train procedure ###
    def train_step(dataset):
        with tf.GradientTape() as tape:
            input_ids, input_pos, masked_lm_positions, masked_lm_ids, masked_lm_weights = dataset
            log_probs = model([input_ids, input_pos, masked_lm_positions], training=True)
            (masked_lm_loss, masked_lm_preds, 
            masked_lm_ids, masked_lm_weights) = get_masked_lm_output_metric_fn(
                log_probs, masked_lm_ids, masked_lm_weights
            )
        grads = tape.gradient(masked_lm_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return masked_lm_loss, masked_lm_preds, masked_lm_ids, masked_lm_weights

    @tf.function
    def distributed_train_step(dataset_inputs):
        loss, preds, ids, weights = strategy.run(train_step, args=(dataset_inputs,))
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        preds = strategy.gather(preds, axis=0)
        ids = strategy.gather(ids, axis=0)
        weights = strategy.gather(weights, axis=0)
        # update loss and metric
        train_loss.update_state(loss)
        train_metric.update_state(
            y_true=ids, 
            y_pred=preds, 
            sample_weight=weights,
        )


    ### evaluation procedure ###
    def infer_step(dataset):
        input_ids, input_pos, masked_lm_positions, masked_lm_ids, masked_lm_weights = dataset
        log_probs = model([input_ids, input_pos, masked_lm_positions], training=False)
        (masked_lm_loss, masked_lm_preds, 
        masked_lm_ids, masked_lm_weights) = get_masked_lm_output_metric_fn(
            log_probs, masked_lm_ids, masked_lm_weights
        )
        return masked_lm_loss, masked_lm_preds, masked_lm_ids, masked_lm_weights

    @tf.function
    def distributed_infer_step(dataset_inputs):
        loss, preds, ids, weights = strategy.run(infer_step, args=(dataset_inputs,))
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        preds = strategy.gather(preds, axis=0)
        ids = strategy.gather(ids, axis=0)
        weights = strategy.gather(weights, axis=0)
        # update loss and metric
        infer_loss.update_state(loss)
        infer_metric.update_state(
            y_true=ids, 
            y_pred=preds, 
            sample_weight=weights,
        )


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
            print(f'Epoch : {epoch+1}/{int(arg.epochs)} '\
                f'Train_Loss : [{float(train_loss.result()):.7f}] Train_Acc : [{float(train_metric.result()):.7f}] '\
                f'Valid_Loss : [{float(infer_loss.result()):.7f}] Valid_Acc : [{float(infer_metric.result()):.7f}]')

            # save model based on evaluation loss
            if best_loss > infer_loss.result():
                best_loss = infer_loss.result()
                best_model = model
                # save bert module weights only (시간 소요 크게 없음)
                with open("/".join(arg.save_dir.split("/")[:-2]) + "/bert_module.pickle", "wb") as f:
                    pickle.dump(best_model.layers[2].get_weights(), f)
                # save total mlm model (시간 소요 조금 있음)
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
        print("Model Weights Loaded")        
        for step_v, v in enumerate(eval_dataset):
            distributed_infer_step(v)
            # show evaluate loss & metric logs per 1 step
            print(f'Steps : {step_v+1}/{arg.total_infer_steps} Test_Loss : {float(infer_loss.result()):.7f} Test_Metric : {float(infer_metric.result()):.7f}'
                  , end='\r')
        # show loss & metric logs per 1 epoch
        print("")
        print(f'Test_Loss : [{float(infer_loss.result()):.7f}] Test_Acc : [{float(infer_metric.result()):.7f}]')

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
