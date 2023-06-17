import pandas as pd
import torch
from transformers import BertTokenizer
from classifier import BertClassifier
from transformers import BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from path import train_path,valid_path,test_path

tokenizer_path = 'cointegrated/rubert-tiny'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model_path = 'cointegrated/rubert-tiny'
model = BertForSequenceClassification.from_pretrained(model_path)
out_features = model.bert.encoder.layer[1].output.dense.out_features
model.classifier = torch.nn.Linear(312, 2)

train_data_nan = pd.read_csv(train_path,sep=";")
train_data=train_data_nan.dropna()
valid_data_nan = pd.read_csv(valid_path,sep=";")
valid_data=valid_data_nan.dropna()
test_data_nan  = pd.read_csv(test_path,sep=";")
test_data=test_data_nan.dropna()

classifier = BertClassifier(
    model_path='cointegrated/rubert-tiny',
    tokenizer_path='cointegrated/rubert-tiny',
    n_classes=2,
    epochs=10,
    model_save_path='/content/bert.pt'
)

classifier.preparation(
    X_train=list(train_data['text']),
    y_train=list(train_data['label']),
    X_valid=list(valid_data['text']),
    y_valid=list(valid_data['label'])
)



classifier.train()


texts = list(test_data['text'])
labels = list(test_data['labels'])
predictions = [classifier.predict(t) for t in texts]


precision, recall, f1score = precision_recall_fscore_support(labels, predictions,average='macro')[:3]

print(f'precision: {precision}, recall: {recall}, f1score: {f1score}')


