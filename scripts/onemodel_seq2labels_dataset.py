import torch
import json
from datasets import Dataset

class Seq2Labels:
    def __init__(self, tokenizer, data, is_roberta, aggregation_mode="first", device="cuda"):
        self.tokenizer = tokenizer
        self.data = data
        self.is_roberta = is_roberta
        self.aggregation_mode = aggregation_mode
        self.device = device
        for key in [
            "input_ids", "subtokens", "labels", "labels_mask", "left_mask", 
            "right_mask", "token_type_ids"
            ]:
            self.data[key] = []
        # if there are no labels
        if "first_tags" not in self.data and "second_tags" not in self.data:
            self.data["first_tags"] = [None for _ in range(len(self.data["tokens"]))]
            self.data["second_tags"] = [None for _ in range(len(self.data["tokens"]))]
        
    def tokenize(self, tokens):
        """
        input: `tokens` -- list of tokens
        returns: tuple of `w_ids` -- LongTensor of word_ids from CLS to the last non-special subtoken
                          `inp_ids` -- LongTensor of input_ids for all subtokens (including BOS/EOS)
                          `subtokens_list`
        """
        if tokens[0] == "<CLS>": # avoiding special tokens
            tokens = tokens[1:]
        if not self.is_roberta: # GPT2TokenizerFast does not add BOS/EOS by default
            tokens = [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
        tokenization = self.tokenizer(tokens, is_split_into_words=True)
        # calculating word_ids from CLS up to the last subtoken (not SEP)
        if self.is_roberta:
            w_ids = torch.LongTensor([0]+[num+1 for num in tokenization.word_ids()[1:-1]]) # CLS/SEP in word_ids == None
        else: # if FRED is used
            w_ids = torch.LongTensor(tokenization.word_ids()[:-1]) # in gpt2tokenizer CLS/SEP word_ids are numbers
        inp_ids = torch.LongTensor(tokenization["input_ids"])
        subtokens_list = [self.tokenizer.decode(inp_id) for inp_id in inp_ids]
        return w_ids, inp_ids, subtokens_list
    
    def extract_mask_and_labels(self, subtokens_num, tokens_num, words_ids, first_tags, second_tags):
        """
        input:  `subtokens_num`
                `tokens_num`
                `words_ids`
                `first_tags` -- list of list of labels for token replacement/deletion
                `second_tags` -- list of list of labels for token insertion
        returns: tuple of   `tags` -- list of labels for sentence tokens and spaces
                            `left_mask`, `right_mask` -- masks for aggregation of len `tags`
                            `token_type_mask` -- mask of len `tags` with `1` for spaces and `0` for tokens
                            `labels_mask` -- mask of len `tags` with True     
        """
        # comprising list of labels, if there are any
        if first_tags is not None and second_tags is not None:
            tags = ['>'.join(second_tags[0])] # the first operation is either `keep` or `insertion`
            assert len(first_tags) == len(second_tags)
            for first_tag, second_tag in zip(first_tags[1:], second_tags[1:]):
                tags.append(first_tag[0])
                tags.append(">".join(second_tag))
        else:
            tags = [None for _ in range(2*tokens_num-1)]
        words_ids_set = list(set(words_ids.tolist()[1:])) # ids of real words
        # calculating mask of left neigbours, which are either first or last subtoken of a real word
        left_mask = torch.cat([torch.LongTensor([torch.where(words_ids == x)[0][0], torch.where(words_ids == x)[0][-1]])
                                            for x in sorted(words_ids_set)]) 
        left_mask = torch.cat([torch.LongTensor([0]), left_mask]) # for CLS
        # calculating mask of right neigbours, which are double first subtokens if aggr_mode is first or first and last if last
        if self.aggregation_mode == "first":
            right_mask = torch.cat([torch.LongTensor([torch.where(words_ids == x)[0][0], torch.where(words_ids == x)[0][0]])
                                            for x in sorted(words_ids_set)])
        elif self.aggregation_mode == "last":
        # adjusting left_mask for last aggr_mode: every even num is for space and corresponds to the last subtoken, 
        # so we make every eneven equal to it 
            left_mask[1::2] = left_mask[2::2]
            right_mask = torch.cat([torch.LongTensor([torch.where(words_ids == x)[0][0], torch.where(words_ids == x)[0][-1]])
                                            for x in sorted(words_ids_set)]) 
        right_mask = torch.cat([right_mask, torch.LongTensor([subtokens_num-1])])  # for SEP 
        assert left_mask.shape[0] == len(tags) == right_mask.shape[0] == 2*tokens_num-1
        # comprising token_type_mask
        token_type_mask = torch.ones_like(right_mask)
        token_type_mask[1::2] = 0 
        # comprising labels_mask
        labels_mask = torch.ones_like(right_mask, dtype=bool)       
        return tags, left_mask, right_mask, token_type_mask, labels_mask
    
    def __call__(self):
        for i in range(len(self.data["tokens"])):
            current_word_ids, current_input_ids, current_subtokens = self.tokenize(self.data["tokens"][i])
            self.data["input_ids"].append(current_input_ids)
            self.data["subtokens"].append(current_subtokens)
            current_tags, current_left_mask, current_right_mask, current_token_type_mask, current_labels_mask =\
                    self.extract_mask_and_labels(subtokens_num=len(current_subtokens), 
                                                 tokens_num=len(self.data["tokens"][i]),
                                                 words_ids=current_word_ids, 
                                                 first_tags=self.data["first_tags"][i],
                                                 second_tags=self.data["second_tags"][i])
            self.data["labels"].append(current_tags)
            self.data["left_mask"].append(current_left_mask)
            self.data["right_mask"].append(current_right_mask)
            self.data["token_type_ids"].append(current_token_type_mask)
            self.data["labels_mask"].append(current_labels_mask)
        self.dataset = Dataset.from_dict(self.data)
        self.dataset.set_format('torch', device=self.device)
        if 'first_tags' in self.dataset.column_names and 'second_tags' in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns(['first_tags', 'second_tags'])
        return self.dataset
            
     