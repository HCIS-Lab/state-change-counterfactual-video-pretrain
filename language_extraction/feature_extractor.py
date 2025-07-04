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
    def __init__(self, split, states_path, mappings_path, workers=4):
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
        self.embed_dir = "language_features/embeddings_FLAVA"
        # self.symlink_dir = "language_features/symlinks_v3"
        self.states_path = states_path #"../states.json"
        # self.mappings_path = mappings_path # "../narration_to_vid_mapping.json"
        # self.previous_states_path = "../states.json"
    
    def get_filename(self, narration):
        assert type(narration) == str
        
        clean_filename =  "".join(x for x in narration if x.isalnum())
        if clean_filename[0].isnumeric():
            clean_filename = '_' + clean_filename

        save_path = os.path.join(self.embed_dir, clean_filename)
        
        # assert not os.path.exists(save_path + '.npy'), "you are overwriting existing features"

        condition = os.path.exists(save_path + '.npy')

        if not condition:
            return save_path
        else:
            i = 0
            while condition:
                i += 1
                condition = os.path.exists(save_path + '_' + str(i) + '.npy')
            return save_path + '_' + str(i)
    
    def inference(self):

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
        if not os.path.exists(self.embed_dir):
            os.mkdir(self.embed_dir)
        # if not os.path.exists(self.symlink_dir):
        #     os.mkdir(self.symlink_dir)

        with open(self.states_path, 'r') as f:
                states = json.loads(f.read())

        # with open(self.mappings_path, 'r') as f:
        #         mappings = json.loads(f.read())
        
        # with open(self.previous_states_path, 'r') as f:
        #         prev_states = json.loads(f.read())

        # assert len(states) == len(mappings)
        
        # not neeeded but doing it to ensure they are in order when saving .npy (see below)
        keys = ['Before', 'After', 'CF 1', 'CF 2', 'CF 3']

        iters = 0
        to_file = 0

        narrations = list(states.keys())
        N = len(narrations)
        # K = len(list(prev_states.keys()))

        assert N == len(states)

        interval_length = N // self.workers
        remainder = N % self.workers
        
        lower = self.split * interval_length
        upper = (self.split + 1) * interval_length

        # for j in range(lower, upper):
        # for j in range(K, N):
        for j in range(N):
            # try:
            k = narrations[j]
    
            v = states[k]
            
            if len(k) > 100:
                text_input = [" ".join(k.split())] + [v[key] for key in keys] # (see here)
                tensor_path = self.get_filename(" ".join(k.split()))
            else:
                text_input = [k] + [v[key] for key in keys] # (see here)
                tensor_path = self.get_filename(k)

            tokens = self.tokenize(text_input)
            embeddings = self.embed(tokens).detach().cpu().numpy()
    

            np.save(tensor_path, embeddings, allow_pickle=True)
            
            # NOTE: we had a symlink mapping initially to speed things up but eventually abandoned it, which is why you see these commented chunks
            
            # for vid in mappings[k]:
            #     save_path = os.path.join(self.symlink_dir, vid)
            #     if os.path.exists(save_path):
            #         print(os.readlink(save_path), " ".join(k.split()), save_path)
            #         print("#############################################################")
            #         continue
            #     # assert not os.path.exists(save_path), "you are overwriting existing symlinks"
            #     os.symlink(os.path.abspath(tensor_path + '.npy'), save_path)
            #     to_file += 1
            
            iters += 1
            if iters % 1000 == 0:
                print("Currently at iter: ", iters)
            # except Exception as e:
            #     print(j)
            #     print(narrations[j])
            #     print(states[narrations[j]])
            #     print([vid for vid in mappings[narrations[j]]])
            #     print(e)
            #     break
            
            # if iters == 9:
            #     break
        
        if self.split == self.workers - 1:
            for j in range(upper, upper + remainder):
                k = narrations[j]
                v = states[k]

                text_input = [k] + [v[key] for key in keys] # (see here)

                tokens = self.tokenize(text_input)
                embeddings = self.embed(tokens).detach().cpu().numpy()

                tensor_path = self.get_filename(k)

                np.save(tensor_path, embeddings, allow_pickle=True)

                for vid in mappings[k]:
                    save_path = os.path.join(self.symlink_dir, vid)
                    assert not os.path.exists(save_path), "you are overwriting existing symlinks"
                    os.symlink(os.path.abspath(tensor_path + '.npy'), save_path)
                    to_file += 1
            
            iters += 1
            if iters % 1000 == 0:
                print("Currently at iter: ", iters)
            
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
    mappings_path = sys.argv[3] #"narration_to_vid_mapping_0.json"
    
    embedder = Embedder(split=split, states_path=states_path, mappings_path="")

    # assert embedder.tokenizer.padding_side == "right", "Bart models' recommended padding side is to the right"
    # assert embedder.tokenizer.pad_token == "<pad>"

    saved_files = embedder.inference()
    
    print()
    print("Created %d files." % (saved_files))
    print()
