from transformer_model import build_transformer 
from dataset import BilingualDataset, causal_mask 
from config import get_config, get_weights_file_path 

import torchtext.datasets as datasets 
import torch
from torch import optim 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR 

import warnings 
from tqdm import tqdm 
import os 
from pathlib import Path
 
# Huggingface datasets and tokenizers 
from datasets import load_dataset 
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 

import torchmetrics
import torchmetrics.text
from torch.utils.tensorboard import SummaryWriter 

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class TransformerLitModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config 
        self.console_width = 80
        
        # Tensorboard 
        self.writer = SummaryWriter(config['experiment_name'])       
        
        try: 
            # get the console window width 
            with os.popen('stty size', 'r') as console: 
                _, console_width = console.read().split() 
                self.console_width = int(console_width) 
        except: 
            # If we can't get the console width, use 80 as default 
            self.console_width = 80 
            
        # Create the directory if it doesn't exist
        save_dir = "weights"
        os.makedirs(save_dir, exist_ok=True)
        
        #Validation variables
        self.val_count = 0 
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 
        self.val_num_examples = 2
        
        #Train variables
        self.train_losses =[] 
        
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
        decoder_input = batch['decoder_input'].to(device) # (B, seq_len) 
        encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len) 
        decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len,-seq_len) 
        
        #with torch.cuda.amp.autocast(enabled=True):
            
        # Run the tensors through the encoder, decoder and the projection layer 
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model) 
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) 
        proj_output = self.model.project(decoder_output) # (B, seq_len, vocab_size) 
        
        # Compare the output with the label 
        label = batch['label'].to(device) # (B, seg_len)
            
        # Compute the loss using a simple cross entropy 
        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)) 
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("loss = ", loss.item(), prog_bar=True) 
        #batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) 

        self.train_losses.append(loss.item())         
        
        # Log the loss 
        self.writer.add_scalar('train,loss', loss.item(), self.trainer.global_step) 
        self.writer.flush() 
        
        # Backpropagate the loss 
        loss.backward(retain_graph=True) 

        return loss


    def validation_step(self, batch, batch_idx):       
        max_len = self.config['seq_len'] 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        if self.val_count == self.val_num_examples:         
            print('-'*self.console_width)
            return 
        
        self.val_count += 1 
        with torch.no_grad():             
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len) 

            # check that the batch size is 1 
            assert encoder_input.size(0) == 1, "Batch  size must be 1 for val"

            model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0] 
            model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy()) 

            self.val_source_texts.append(source_text) 
            self.val_expected.append(target_text) 
            self.val_predicted.append(model_out_text) 

            # Print the source, target and model output             
            print('-'*self.console_width) 
            print(f"{f'SOURCE: ':>12}{source_text}") 
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")  
            #print('-'*self.console_width)
            
            
    def on_validation_epoch_end(self):
        writer = self.writer
        if writer:
            # Evaluate the character error rate 
            # Compute the char error rate 
            metric = torchmetrics.text.CharErrorRate() 
            cer = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('validation cer', cer, self.trainer.global_step) 
            self.log("val/cer", cer, prog_bar=True)
            print(f'Val CER at end of epoch {self.trainer.current_epoch} = {cer}')
            writer.flush() 

            # Compute the word error rate 
            metric = torchmetrics.text.WordErrorRate() 
            wer = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('validation wer', wer, self.trainer.global_step)
            self.log("val/wer", wer, prog_bar=True)
            print(f'Val WER at end of epoch {self.trainer.current_epoch} = {wer}')
            writer.flush() 

            # Compute the BLEU metric 
            metric = torchmetrics.text.BLEUScore() 
            bleu = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('validation BLEU', bleu, self.trainer.global_step)
            self.log("val/bleu", bleu, prog_bar=True)
            print(f'Val BLEU at end of epoch {self.trainer.current_epoch} = {bleu}')
            writer.flush() 
            
        self.val_count = 0
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 

    def test_step(self, batch, batch_idx):
        pass
    
      
    def on_train_epoch_end(self):
        # Save the model at the end of every epoch   
        mean_loss = sum(self.train_losses) / len(self.train_losses)
        print(f'Mean training loss at end of epoch {self.trainer.current_epoch} = {mean_loss}')
        model_filename = get_weights_file_path(self.config, f"{self.trainer.current_epoch:02d}") 
        torch.save({ 
                    'epoch': self.trainer.current_epoch, 
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(), 
                    'global_step': self.trainer.global_step}
                   , model_filename) 
        self.train_losses = []
            
            
    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device): 
    
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step 
        encoder_output = model.encode(source, source_mask) 

        # Initialize the decoder input with the sos token 
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) 

        while True: 
            if decoder_input.size(1) == max_len:  
                break 

            # build mask for target 
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) 

            # calculate output 
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) 

            # get next token 
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
                ], dim = 1
            )

            if next_word == eos_idx: 
                break 

        return decoder_input.squeeze(0)
    
    def configure_optimizers(self): 
        suggested_lr = 10E-03
        
        steps_per_epoch = len(self.train_dataloader())
        scheduler_dict = {
            "scheduler":  optim.lr_scheduler.OneCycleLR(
        self.optimizer, max_lr=suggested_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=self.trainer.max_epochs, 
        pct_start=10/self.trainer.max_epochs,
        three_phase=True,
        div_factor=10,
        final_div_factor=10,
        anneal_strategy='cos',
            ),
            "interval": "step",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict} # 
    
    
    ####################
    # DATA RELATED HOOKS
    ####################
    
    def get_all_sentences(self, ds, lang): 
        for item in ds: 
            yield item['translation'][lang]
    
    def get_model(self, config, vocab_src_len, vocab_tgt_len): 
        model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], 
                                  config["seq_len"], d_model=config["d_model"])
        return model
    
    def get_or_build_tokenizer(self, config, ds, lang): 
        tokenizer_path = Path(config['tokenizer_file'].format(lang)) 
        if not Path.exists(tokenizer_path): 
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour 
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) 
            tokenizer.pre_tokenizer = Whitespace() 
            trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path)) 
        else: 
            tokenizer = Tokenizer.from_file(str(tokenizer_path)) 
        return tokenizer

    def prepare_data(self):  
        config = self.config
        # download
        # It only has the train split, so we divide it overselves 
        ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')        
        
        # Define a function to filter dataset as per assignment requirement
        def filter_examples(example):
            source_text = example['translation'][config['lang_src']]
            target_text = example['translation'][config['lang_tgt']]
            return len(source_text) <= 150 and len(target_text) <= len(source_text) + 10

        # Filter the dataset based on the custom filter function
        ds_raw = ds_raw.filter(filter_examples)
        
        # Build tokenizers 
        self.tokenizer_src = self.get_or_build_tokenizer(config, ds_raw, config['lang_src'])
        self.tokenizer_tgt = self.get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
        
        #Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        
        # Get model        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.model = self.get_model(config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size()).to(device)
        
        #Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], eps=1e-9) 
        
        #Preload model
        if config['preload']: 
            model_filename = get_weights_file_path(config, config['preload']) 
            print(f'Preloading model {model_filename}') 
            state = torch.load(model_filename) 
            self.model.load_state_dict(state['model_state_dict'])
            self.trainer.global_step = state['global_step']
            print("Preloaded")
            
      
        # Keep 90% for training, 10% for validation 
        train_ds_size = int(0.9* len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        self.train_ds = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, config['lang_src'], 
                                    config['lang_tgt'], config['seq_len'])
        self.val_ds = BilingualDataset(val_ds_raw, self.tokenizer_src, self.tokenizer_tgt, config['lang_src'], 
                                    config['lang_tgt'], config['seq_len'])

        # Find the maximum length of each sentence in the source and target sentence 
        max_len_src = 0 
        max_len_tgt = 0 

        for item in ds_raw: 
            src_ids = self.tokenizer_src.encode(item['translation'][config['lang_src']]).ids
            tgt_ids = self.tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f'Max length of source sentence: {max_len_src}') 
        print(f'Max length of target sentence: {max_len_tgt}') 
  

    def setup(self, stage=None):
        pass 
       

    def train_dataloader(self):                   
        return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=True) 
    

    def test_dataloader(self):
        pass
    
    def collate_fn(self, batch):
            #print('----------------------------------------------')
            encoder_input_max = max(x["encoder_str_length"] for x in batch)
            decoder_input_max = max(x["decoder_str_length"] for x in batch)
            #print('----------------------------------------------')
            #print('encoder_input_max = ',encoder_input_max)
            #print('decoder_input_max = ',decoder_input_max)
            #print('----------------------------------------------')
            encoder_inputs = []
            decoder_inputs = []
            encoder_mask = []
            decoder_mask = []
            label = []
            src_text = []
            tgt_text = []

            for b in batch:
                encoder_inputs.append(b["encoder_input"][:encoder_input_max])
                decoder_inputs.append(b["decoder_input"][:decoder_input_max])
                encoder_mask.append((b["encoder_mask"][0, 0, :encoder_input_max]).unsqueeze(0).unsqueeze(0).unsqueeze(0).int())
                decoder_mask.append((b["decoder_mask"][0, :decoder_input_max, :decoder_input_max]).unsqueeze(0).unsqueeze(0).int())
                label.append(b["label"][:decoder_input_max])
                src_text.append(b["src_text"])
                tgt_text.append(b["tgt_text"])
            return {
              "encoder_input":torch.vstack(encoder_inputs),
              "decoder_input":torch.vstack(decoder_inputs),
              "encoder_mask": torch.vstack(encoder_mask),
              "decoder_mask": torch.vstack(decoder_mask),
              "label":torch.vstack(label),
              "src_text":src_text,
              "tgt_text":tgt_text
            }
