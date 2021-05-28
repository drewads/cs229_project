import numpy as np
import tensorflow as tf
from tensorflow import keras
import img_proc
import evaluate
from DLNN import DLNN

def extract_features(base_model, train_data):
    for sample in train_data:
        print(sample)
        print("Before")
        features = base_model.predict(train_data, steps=100)
        # TODO: normalize and flatten features vector
        print("HELLO")
        print(np.shape(features), np.shape(sample[0]), np.shape(sample[1]))
        yield (features, sample[1])

def transfer(train_data, BATCH_SIZE=100):
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False)
    feats = extract_features(base_model, train_data)
    #model = DLNN(feats, img_proc.num_examples('data/train_sep'), img_proc.num_features(), [1024,256,64,16,4,1], epochs = 20)
    model = DLNN(feats, [1024, 256, 64, 16, 4, 1], epochs=20)
    y_train_pred = model.predict(train_data, steps=100)
    y_train = train_data.get_labels()

    data_gen_valid = img_proc.Data_Generator('data/valid', BATCH_SIZE, shuffle=False, flatten=True)
    y_valid = data_gen_valid.get_labels()
    y_valid_pred = model.predict(data_gen_valid)

    auc_roc,threshold_best = evaluate.ROCandAUROC(y_valid_pred,y_valid,'ROC_valid_data_dlnn.jpeg')

    print(f"\nArea Under ROC = {auc_roc}")
    tp,fn,fp,tn = evaluate.counts(y_train_pred, y_train, threshold = threshold_best)
    acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    print("\nStats for predictions on train set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")

    tp,fn,fp,tn = evaluate.counts(y_valid_pred, y_valid, threshold = threshold_best)
    acc,prec,sens,spec,F1 = evaluate.stats(tp,fn,fp,tn)
    print("\nStats for predictions on validation set:")
    print(f"Threshold = {threshold_best}")
    print(f"Accuracy = {acc}")
    print(f"Precision = {prec}")
    print(f"Sensitivity = {sens}")
    print(f"Specificity = {spec}")
    print(f"F1 score = {F1}")




def main():
    #TODO: also need to pass normalization function
    BATCH_SIZE = 100
    train_data = img_proc.Data_Generator('data/train_sep', BATCH_SIZE, shuffle=True, flatten=True)
    print("HELLO")
    print(type(train_data))
    transfer(train_data)


if __name__ == "__main__":
    main()


