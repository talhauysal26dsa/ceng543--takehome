# Sentiment Analysis - Full Model Comparison

*Evaluated 4 models*

---

## LSTM_BERT

### Test Set Performance
- **Accuracy:** 0.9121 (91.21%)
- **Macro F1:** 0.9121
- **F1 Negative:** 0.9126
- **F1 Positive:** 0.9116
- **Precision Negative:** 0.9074
- **Precision Positive:** 0.9169
- **Recall Negative:** 0.9178
- **Recall Positive:** 0.9063

### Training Information
- **Total Epochs:** 5
- **Best Val Accuracy:** 0.9121
- **Avg Time/Epoch:** 649.70s
- **Convergence (85%):** 1 epochs

### Confusion Matrix
```
              Predicted
              Neg    Pos
Actual  Neg  [11473   1027]
        Pos  [ 1171  11329]
```

---

## GRU_BERT

### Test Set Performance
- **Accuracy:** 0.9112 (91.12%)
- **Macro F1:** 0.9112
- **F1 Negative:** 0.9108
- **F1 Positive:** 0.9116
- **Precision Negative:** 0.9149
- **Precision Positive:** 0.9076
- **Recall Negative:** 0.9068
- **Recall Positive:** 0.9157

### Training Information
- **Total Epochs:** 5
- **Best Val Accuracy:** 0.9112
- **Avg Time/Epoch:** 661.52s
- **Convergence (85%):** 1 epochs

### Confusion Matrix
```
              Predicted
              Neg    Pos
Actual  Neg  [11335   1165]
        Pos  [ 1054  11446]
```

---

## GRU_GLOVE

### Test Set Performance
- **Accuracy:** 0.8772 (87.72%)
- **Macro F1:** 0.8772
- **F1 Negative:** 0.8789
- **F1 Positive:** 0.8755
- **Precision Negative:** 0.8672
- **Precision Positive:** 0.8879
- **Recall Negative:** 0.8910
- **Recall Positive:** 0.8635

### Training Information
- **Total Epochs:** 7
- **Best Val Accuracy:** 0.8772
- **Avg Time/Epoch:** 56.77s
- **Convergence (85%):** 1 epochs

### Confusion Matrix
```
              Predicted
              Neg    Pos
Actual  Neg  [11137   1363]
        Pos  [ 1706  10794]
```

---

## LSTM_GLOVE

### Test Set Performance
- **Accuracy:** 0.8616 (86.16%)
- **Macro F1:** 0.8615
- **F1 Negative:** 0.8584
- **F1 Positive:** 0.8646
- **Precision Negative:** 0.8786
- **Precision Positive:** 0.8461
- **Recall Negative:** 0.8392
- **Recall Positive:** 0.8840

### Training Information
- **Total Epochs:** 8
- **Best Val Accuracy:** 0.8616
- **Avg Time/Epoch:** 225.95s
- **Convergence (85%):** 2 epochs

### Confusion Matrix
```
              Predicted
              Neg    Pos
Actual  Neg  [10490   2010]
        Pos  [ 1450  11050]
```

---

