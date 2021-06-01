import numpy as np
import tensorflow as tf
from tensorflow import keras
import img_proc
import evaluate
from pathlib import Path
from DLNN import DLNN

# def extract_features(base_model, train_data):
#     for sample in train_data:
#         print(sample)
#         print("Before")
#         features = base_model.predict(train_data, steps=100)
#         # TODO: normalize and flatten features vector
#         print("HELLO")
#         print(np.shape(features), np.shape(sample[0]), np.shape(sample[1]))
#         yield (features, sample[1])

def transfer(data_dir, BATCH_SIZE=100):
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        input_shape=(224,224,3),
        include_top=False)
    feats = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=True, flatten=False, model=base_model, flatten_post_model=True)
    model = DLNN(feats, [1024, 256, 64, 16, 4, 1], epochs=1)
    model.save('saved_DLNN_transfer')
    data_gen_train_test = img_proc.Data_Generator(data_dir / 'train_sep', BATCH_SIZE, shuffle=False, flatten=False, model=base_model, flatten_post_model=True)
    y_train_pred = model.predict(data_gen_train_test)
    y_train = data_gen_train_test.get_labels()

    data_gen_valid = img_proc.Data_Generator(data_dir / 'valid', BATCH_SIZE, shuffle=False, flatten=False, model=base_model, flatten_post_model=True)
    y_valid = data_gen_valid.get_labels()
    y_valid_pred = model.predict(data_gen_valid)

    auc_roc,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_transfer.jpeg')

    print(f"\nArea Under ROC = {auc_roc}")
    tp,fn,fp,tn = evaluate.counts(y_train_pred, y_train)
    acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    print("\nStats for predictions on train set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")

    tp,fn,fp,tn = evaluate.counts(y_valid_pred, y_valid)
    acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    print("\nStats for predictions on validation set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")

def main():
    BATCH_SIZE = 100
    DATA_DIR = Path('data_200')
    transfer(DATA_DIR, BATCH_SIZE)

if __name__ == "__main__":
    main()


