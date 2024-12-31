# ASL Fingerspelling Recognition with MobileNetV2

This repository implements an American Sign Language (ASL) fingerspelling recognition system using PyTorch and the MobileNetV2 architecture. The model classifies images of ASL fingerspelling gestures into corresponding letters.

## **Project Overview**
The project is structured as a modular pipeline for:
1. Dataset preprocessing and splitting.
2. Training and validating a MobileNetV2-based classifier.
3. Visualizing training metrics such as loss and accuracy.

---

## **Dataset**
The ASL dataset is sourced from [Kaggle's ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet). It contains images of 26 letters in the ASL alphabet and 3 special commands.

```bash
pip install -r requirements.txt