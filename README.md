# Code for the paper **"Grammatical Error Correction via Sequence Tagging for Russian"**
* ## Preprocessing
  To preprocess data from RULEC-GEC, RU-Lang8 or GERA, use
  ```
  python preprocess.py -d DATA_CONFIGS -V -o OUTPUT_FILE
  ```

  * `DATA_CONFIGS` -- configuration files. Configuration files, in their turn, should contain paths to m2-files, parsed initial sentences and parsed corrected sentences (we adopted DeepPavlov as a morphological parser).

    An example configuration file:
    ```
    [paths]
    data_path = /data/
    samplename = GERA
    data = ${data_path}${samplename}.test.m2
    parsed_data = ${data_path}morph_files_for_preprocession/Parsed_${samplename}.test.txt
    parsed_correct_data = ${data_path}morph_files_for_preprocession/Parsed_correct${samplename}.test.txt
    ```
  * `-V` -- optional, provides verification of the assigned labels -- if correct sentences may be inferred from the initial ones and the given labels.
  * `OUTPUT_FILE` -- by default, "OUTPUT.txt".
    
    It contains sentences in the format `<token>\t<token_tag>><insertion_tag(s)>`, e.g.
    ```
    <CLS>	Keep>Keep
    Конечно	Keep>Insert,
    были	Keep>Keep
    исторические	Keep>Keep
    произведения	Keep>Keep
    и	Keep>Keep
    до	Keep>Keep
    этого	Keep>Keep
    .	Keep>Keep
    ```
* ## Training
  To train or finetune the model, use
  ```
  python onemodel_main.py -t TRAIN_SAMPLES -v VAL_SAMPLES -o OUTPUT_DIR -l LABELS2ID_FILE -F -L --LR -B BATCH_SIZE -A AGGREGATION_TYPE -M MODE -T -G ACCUM_STEP -W WEIGHTS -E N_EPOCHS
  ```
  * `TRAIN_SAMPLES` -- preprocessed training samples
  * `VAL_SAMPLES` -- preprocessed validation samples
  * `OUTPUT_DIR` -- a folder to save weights to
  * `LABELS2ID_FILE` -- dictionary of ids mapped to labels (needed in the case of finetuning)
  * `-F` -- if `FRED-T5-1.7B` should be used as an encoder model (ruRoberta is the default encoder)
  * `LR` -- learning rate, the default value is 1e-5
  * `BATCH_SIZE` -- default=16
  * `AGGREGATION_TYPE` -- defines whether the token would be represented by its first or last embedding, the default value is "first"
  * `MODE` -- "train" (default) or "finetune"
  * `-T` -- if token type embeddings should be added
  * `ACCUM_STEP` -- gradient accumulation step, default=1
  * `WEIGHTS` -- model weights (needed in the case of finetuning)
  * `N_EPOCHS` -- number of training epochs, the default value is 2
    
* ## Inference
  For label prediction, use
  ```
  python cls_inference.py -i INPUT_FILE -o OUT_FILE -l LABELS2ID_FILE -B BATCH_SIZE -W WEIGHTS -H HYPERPARAMETERS_CONFIG
  ```
  * `INPUT_FILE` -- should be in the preprocessed format.
  
  If your file is a list of sentences, use `python make_preprocessed.py -i INPUT_FILE -o OUTPUT_FILE` to transform it.
  * `OUT_FILE` -- by default, "preds" 
  * `LABELS2ID_FILE` - dictionary of ids mapped to labels
  * `BATCH_SIZE` -- default=16
  * `WEIGHTS` -- model weights
  * `HYPERPARAMETERS_CONFIG` -- the configuration file which is automatically saved as a result of training/finetuning
* ## Decoding
  To transform the tokens according to predicted labels, use
  ```
  python decode.py -p PREDS_FILE -f FEATS_FILE -o OUT_FILE -J JSON_CONFIG
  ```
  * `PREDS_FILE` -- file with predicted labels (the result of model inference)
  * `FEATS_FILE` -- parsed initial sentences
  * `OUT_FILE` -- by default, "OUTPUT"
  * `JSON_CONFIG` -- json configuration file to access YandexGPT model:
  ```
  {
    "folder_id": "<>",
    "iam_token": "<>"
  }
  ```
