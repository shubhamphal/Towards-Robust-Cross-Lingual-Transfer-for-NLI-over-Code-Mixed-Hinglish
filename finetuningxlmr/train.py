import config
import dataset
import engine
from model import roberta_model
import data_augmentation
import predict


from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from transformers import get_linear_schedule_with_warmup, AdamW
import torch
from torch import nn, optim
from collections import defaultdict
import pandas as pd
import numpy as np




def run():
    print ("Importing the datasets...\n")
    df = pd.read_csv(config.Training_file)
    
    print ("Imported...\n")

    print ("Processing and doing Data Augmentation ...\n")
    
    print ("back translation started...\n")
    
    
    
    new_df = pd.DataFrame() 
    if config.target_language == 'hindi':
        new_df[['premise','hypothesis','labels']] = df[['hindi_premise','hindi_hypothesis','labels']]
    elif config.target_language == 'english':
        new_df[['premise','hypothesis','labels']] = df[['english_premise','english_hypothesis','labels']]
    elif config.target_language == 'hinglish':
        new_df[['premise','hypothesis','labels']] = df[['premise','hypothesis','labels']]
    else:
        new_df = df
    df_train = new_df
    
    #df =df[['premise','hypothesis','labels']]
    #df_xnli = pd.read_csv(config.XNLI_file)
    #df_xnli = df_xnli[['premise','hypothesis','labels']]   
    #df = pd.concat([df, df_xnli], ignore_index = True)
    print("Training data of size ",df.shape)
    #df = df_xnli
    print("Train Dataframe")
    print(df_train[['premise','hypothesis','labels']].head())
   
    # Multi-Genre NLI Corpus
    #df_mnli = data_augmentation.load_mnli()

    # Cross-Lingual NLI Corpus
    #df_xnli = data_augmentation.load_xnli()

    #df.drop(columns = ['id'], inplace = True)
    #df_train = data_augmentation.proc(df_train)
    
    df = pd.read_csv(config.Testing_file)
    new_df = pd.DataFrame() 
    if config.target_language == 'hindi':
        new_df[['premise','hypothesis','labels']] = df[['hindi_premise','hindi_hypothesis','labels']]
    elif config.target_language == 'english':
        new_df[['premise','hypothesis','labels']] = df[['english_premise','english_hypothesis','labels']]
    elif config.target_language == 'hinglish':
        new_df[['premise','hypothesis','labels']] = df[['premise','hypothesis','labels']]
    else:
        new_df = df
    
    df_test = new_df
    

   

    #df = pd.concat([df, df_mnli, df_xnli], ignore_index = True)

    # Shuffling the dataframe
    df_train = resample(df_train, random_state = config.RANDOM_SEED)
    df_train['premise'] = df_train['premise'].astype(str)
    df_train['hypothesis'] = df_train['hypothesis'].astype(str)
    df_test['premise'] = df_test['premise'].astype(str)
    df_test['hypotheis'] = df_test['hypothesis'].astype(str)
    
    #df_train.drop_duplicates(subset = ['premise', 'hypothesis'], inplace = True)
    #df_test.drop_duplicates(subset = ['premise', 'hypothesis'], inplace = True)
    
    combined_thesis = df_train[['premise', 'hypothesis']].values.tolist()
    combined_thesis_test = df_test[['premise', 'hypothesis']].values.tolist()
    
    df_train['combined_thesis'] = combined_thesis
    df_test['combined_thesis'] = combined_thesis_test
    
    df_train, df_val = train_test_split(
        df_train,
        test_size = 0.1,
        random_state = 0
        )
    
    
    
    
    
    
#     df_test, df_val = train_test_split(
#         df_test,
#         test_size = 0.4,
#         random_state = 0
#         )
    
    print("Test DF")
    print(df_test.isna().sum())
    df_test.dropna(inplace=True)
    
    
    train_data_loader = dataset.create_data_loader(df_train, config.tokenizer, config.max_len, config.batch_size)
    test_data_loader = dataset.create_data_loader(df_test, config.tokenizer, config.max_len, config.batch_size)
    val_data_loader = dataset.create_data_loader(df_val, config.tokenizer, config.max_len, config.batch_size)
    
    print ("Processing and Data Augmentation Done...\n")

    # we save the whole model of roberta_model in the model var which is further used in the code
    device = torch.device(config.DEVICE)
    model = roberta_model(2)
    #model.load_state_dict(torch.load(config.model_load_path),strict=False)
    model= nn.DataParallel(model)
    torch.cuda.empty_cache()
    model = model.to(device)
    #optimizer, scheduler and loss_fn used in the train_roberta_epoch and eval_roberta
    
    
    if config.mode != 'eval':
        optimizer = AdamW(model.parameters(), lr=5e-6, correct_bias=True)
        total_steps = len(train_data_loader) * config.EPOCHS

        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=0,
          num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(device)
        history = defaultdict(list)
        best_accuracy = 0
    
        print ("Training...\n")
        for epochs in range(config.EPOCHS):
            print ('Epoch {} \n'.format(epochs + 1))
            print ("-"*100)
            train_acc, train_loss = engine.train_roberta_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train)
                )
            print ('train_acc {} train_loss {}'.format(train_acc, train_loss))
            val_acc, val_loss = engine.eval_roberta(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(df_val)
                ) 
            print ('val_acc {} val_loss {}'.format(val_acc, val_loss))
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
        
            if val_acc > best_accuracy:
                # here model.state_dict() will save the model and optimizer's parameter
                torch.save(model.state_dict(), config.model_save_path)
                best_accuracy = val_acc
        print ("Training completed...")
        print("Best Accuracy Train",best_accuracy)
        print ("Testing...\n")
    #model.load_state_dict(torch.load(config.model_path),strict=False)
    # predictions
    print('Evaluating predictions')
    y_thesis, y_pred, y_pred_probs, y_test = predict.get_predictions(
    model,
    test_data_loader,
    device
    )
    # Classification_report
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    ln = []
    # tn, fp, fn, tp 
    ln = confusion_matrix(y_test, y_pred).ravel()
    print ("True Negative, False Positive, False Negative, True positive : ", ln)
    print ()
    
    # Accuracy
    print (accuracy_score(y_test, y_pred))
    print()

    print ("DONE!!")
        
if __name__ == '__main__':
    run()
