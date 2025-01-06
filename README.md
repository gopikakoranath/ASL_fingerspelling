# ASL Fingerspelling Recognition in Raspberry Pi with MobileNetV2

This repository implements an American Sign Language (ASL) fingerspelling recognition system using TensorFlow and the MobileNetV2 architecture. The model classifies images of ASL fingerspelling gestures into corresponding letters.

The model is then deployed in a raspberry pi with camera module to aid real-time fingerspelling.

## **Project Overview**

The project repository is structured as a modular pipeline for:
1. Dataset preprocessing and splitting.
2. Training and validating a MobileNetV2-based classifier.
3. Visualizing training metrics such as loss and accuracy.

Notes: 
1. The model was trained and validated in kaggle. The training notebook can be fould in notebooks/training_notebook.ipynb.
2. We had initially implemented in pytorch, but we were unsuccessful in converting torch model to tflite which is more preferrred for edge devices. The pytorch version can be found in pytorch_version/.
---

## **Dataset**
The ASL dataset is sourced from [Kaggle's ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet). It contains images of 26 letters in the ASL alphabet and 3 special commands.

```bash
pip install -r requirements.txt
```

---

## **Model Training**
1. Loaded a pretrained MobileNetV2 model.
2. Added batch normalization layer, dropput regularization and early stopping to reduce overfitting.
4. We are saving the model checkpoint after each epoch.


