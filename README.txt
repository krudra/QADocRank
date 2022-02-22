This repository contains codes for joint learning of query-document/ query-passage pairs

foldwise_train.py
It will take INPUT_PATH, MODEL_PATH, and OUTPUT_PATH as inputs

INPUT_PATH: Should contain foldwise train, validation, and test sets.

File format
Label\tquery_id\tpassage_id\tquery_text\tpassage_text
1\t439\tD25+1\tQ\tD

MODEL_PATH: Path where models will be saved

OUTPUT_PATH: Path where results will be saved

trecdl_train.py

TRAIN_PATH: Should contain training data
TEST_PATH: Contain the test data
OUTPUT_PATH: Contain the result
MODEL_PATH: Path where models will be saved
