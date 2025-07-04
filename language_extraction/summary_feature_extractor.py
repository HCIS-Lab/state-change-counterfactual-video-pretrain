import torch
import numpy as np
import os
import json
import sys
from transformers import AutoTokenizer, FlavaTextModel

print("Cuda available: ", torch.cuda.is_available())

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

class Embedder():
    def __init__(self, split, states_path, workers=4):
        super(Embedder, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
        self.model = FlavaTextModel.from_pretrained("facebook/flava-full").to(device)

        assert ( (split >= 0) and (split <= workers - 1) )
        assert ( (workers >= 1) and (workers <= 8) )

        self.split = split
        self.workers = workers
        self.root_dir = "language_features"
        self.embed_dir = "language_features/summary_embeddings_FLAVA"

        self.states_path = states_path #"../states.json"
    
    def get_filename(self, narration):
        assert type(narration) == str
        
        clean_filename =  "".join(x for x in narration if x.isalnum())
        if clean_filename[0].isnumeric():
            clean_filename = '_' + clean_filename

        save_path = os.path.join(self.embed_dir, clean_filename)
        
        # assert not os.path.exists(save_path + '.npy'), "you are overwriting existing features"

        condition = os.path.exists(save_path[:245] + '.npy')

        if not condition:
            return save_path[:245]
        else:
            i = 0
            while condition:
                i += 1
                condition = os.path.exists(save_path + '_' + str(i) + '.npy')
            return save_path[:245] + '_' + str(i)
    
    def inference(self):

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
        if not os.path.exists(self.embed_dir):
            os.mkdir(self.embed_dir)

        with open(self.states_path, 'r') as f:
                states = json.loads(f.read())
        
        parent_keys = ["order", "key"] # first order then key
        # not neeeded but doing it to ensure they are in order when saving .npy (see below)
        keys = ['CF ' + str(i) for i in range(1,11)]

        iters = 0
        to_file = 0

        narrations = list(states.keys())
        N = len(narrations)

        assert N == len(states)

        skipped_vids = {"file": self.states_path}
        for j in range(N):
            try:
                k = narrations[j]
                v = states[k]

                # first order then key
                cfs = [ [v[parent_k][key] for key in keys] for parent_k in parent_keys]
                
                if len(k) > 200:
                    text_input = [" ".join(k.split())] + cfs[0] + cfs[1] # (see here)
                    tensor_path = self.get_filename(" ".join(k.split()))
                else:
                    text_input = [k] + cfs[0] + cfs[1] # (see here)
                    tensor_path = self.get_filename(k)

                tokens = self.tokenize(text_input)
                embeddings = self.embed(tokens).detach().cpu().numpy()

                np.save(tensor_path, embeddings, allow_pickle=True)
                
                iters += 1
                if iters % 1000 == 0:
                    print("Currently at iter: ", iters)
            
            except Exception as e:
                k = narrations[j]
                print(j)
                skipped_vids[str(j)] = str(j)
                # continue
        
        # if self.split == self.workers - 1:
        #     for j in range(upper, upper + remainder):
        #         k = narrations[j]
        #         v = states[k]

        #         text_input = [k] + [v[key] for key in keys] # (see here)

        #         tokens = self.tokenize(text_input)
        #         embeddings = self.embed(tokens).detach().cpu().numpy()

        #         tensor_path = self.get_filename(k)

        #         np.save(tensor_path, embeddings, allow_pickle=True)

        #         for vid in mappings[k]:
        #             save_path = os.path.join(self.symlink_dir, vid)
        #             assert not os.path.exists(save_path), "you are overwriting existing symlinks"
        #             os.symlink(os.path.abspath(tensor_path + '.npy'), save_path)
        #             to_file += 1
            
            iters += 1
            if iters % 1000 == 0:
                print("Currently at iter: ", iters)
        
        with open("skipped_vids_FLAVA.json", 'w') as f: # we had 1-2 vids that we had to skip due to annotation error --- whole csv embedded into a summary narration
            json.dump(skipped_vids, f)

        return to_file

    def tokenize(self, input_str):
        """
        input: pyhon list of strings
        output: dictionary with batched tensors
        """
        # padding due to batched inputs
        tokens = self.tokenizer(input_str, padding=True, return_tensors="pt").to(self.device)

        return tokens

    def embed(self, tokens):
        """
        input: dictionary with batched tensors (from tokenizer)
        output: batched tensor
        """

        outputs = self.model(**tokens)
        embedding = outputs.last_hidden_state
        # print(embedding.shape) # [6, L, 768] note that L varies per mini-batch

        return embedding
    

if __name__ == '__main__':
    
    assert len(sys.argv) > 1, "enter a partition of the data to embed, default range is [0,3]"
    
    split = int(sys.argv[1])
    
    states_path = sys.argv[2] #"../narration/states_refine_0_0.json"
    
    embedder = Embedder(split=split, states_path=states_path)

    # assert embedder.tokenizer.padding_side == "right", "Bart models' recommended padding side is to the right"
    # assert embedder.tokenizer.pad_token == "<pad>"

    saved_files = embedder.inference()
    
    print()
    print("Created %d files." % (saved_files))
    print()
