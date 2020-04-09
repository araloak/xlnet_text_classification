from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras_radam import RAdam
from keras_bert.layers import Extract
import keras

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_xlnet(args):
    # Load pretrained model
    model = load_trained_model_from_checkpoint(
        config_path=args.config_path,
        checkpoint_path=args.model_path,
        batch_size=args.batch_size,
        memory_len=0,
        target_len=args.maxlen,
        in_train_phase=False,
        attention_type=ATTENTION_TYPE_BI,
    )

    # Build classification model
    last = model.output
    extract = Extract(index=-1, name='Extract')(last)
    output = keras.layers.Dense(units=args.nclass, activation='softmax', name='Softmax')(extract)
    model = keras.models.Model(inputs=model.inputs, outputs=output)
    model.summary()


    # Fit model
    model.compile(
        optimizer=RAdam(args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print(model.summary())
    return model