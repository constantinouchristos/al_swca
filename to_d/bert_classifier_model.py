import os
import torch.nn as nn
import torch.nn.functional as F




class Bert_classifier_adversarial(nn.Module):
    
    def __init__(self,
                 bert_seq_class=None,
                 hidden_dim=768, 
                ):
        
        super(Bert_classifier_adversarial, self).__init__()
        self.bert = bert_seq_class
        
        self.mlp_h = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, inputs,return_h=False):
        
        inputs_bert = {i:j for i,j in inputs.items() if i != 'labels'}
        
        # bert ouput and last hidden state
        bert_sequence_output,hidden_states = self.bert(**inputs_bert)
        

        # hidden states corresponding to CLS token
        first_token_tensor = hidden_states[:, 0]
        
        z_i =  F.relu(self.mlp_h(first_token_tensor))
        
        if return_h:
            return bert_sequence_output,h,z_i
        else:
            return bert_sequence_output,z_i
        
        
    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'weights.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        
    