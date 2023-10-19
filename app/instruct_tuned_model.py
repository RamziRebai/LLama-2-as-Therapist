from torch import bfloat16, LongTensor, cuda
import transformers
import torch
from transformers import LogitsProcessor,LogitsProcessorList


# bit and byte configuration
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16,  # Computation type
    load_in_8bit=False
)
model_id= "RamziRebai/llama-2-7b-therapist-v4"
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# Llama 2 Tokenizer
tokenizer =transformers.AutoTokenizer.from_pretrained(model_id)

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()
print(f"Model {model_id} loaded on {device}")

class EosTokenRewardLogitProcess(LogitsProcessor):
  def __init__(self,eos_token_id: int, max_length: int):

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        if not isinstance(max_length, int) or max_length < 1:
          raise ValueError(f"`max_length` has to be a integer bigger than 1, but is {max_length}")

        self.eos_token_id = eos_token_id
        self.max_length=max_length

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    cur_len = input_ids.shape[-1]
    # start to increese the reward of the  eos_tokekn from 80% max length  progressively on length
    for cur_len in (max(0,int(self.max_length*0.8)), self.max_length ):
      ratio = cur_len/self.max_length
      num_tokens = scores.shape[1] # size of vocab
      scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] =\
       scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]*ratio*10*torch.exp(-torch.sign(scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]))
      scores[:, self.eos_token_id] = 1e2 * ratio
    return scores

logits_process_list= LogitsProcessorList([EosTokenRewardLogitProcess(eos_token_id=tokenizer.eos_token_id, max_length=512)] )

# prompt = "I'm very depressed. How do I find someone to talk to?"
# prompt="When I go to school, I feel like everyone is judging me, even my friends. I get overwhelmed which these thoughts and sometimes cannot get out of what I call a deep hole of thoughts. I barely go to any of our school dances because of all of the people. Not even when I am completely alone do these thoughts go away. I still feel like people can see me and are judging me."
# prompt="I have problems with self esteem, how could you help me?"
# prompt="I am Chris,I am suffering from excessive worry about issues and situations that happen in my experience every day"
# prompt="Hi, My name is Ramzi.I feel depressed"
pipe = transformers.pipeline(model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    #stopping_criteria=stopping_criteria,  # without this model rambles during chat
    logits_processor=logits_process_list,
    max_new_tokens=512,  # max number of tokens to generate in the output
    temperature=0,
    repetition_penalty=1.1
    )


def get_response(text):
  prompt_behavior=f"<s>[INST]Act like a Therapist advisor.You will be provided with a patient request.Answer it appropriately with a helpful tone.Do not respond without your personal informations like names, places or websites.Patient Request: {text} [/INST]"
  leng=len(prompt_behavior)
  result = pipe(prompt_behavior)
  index=result.find("\n")
  result=result[0]['generated_text'][leng:index-1]
  if "\t" in  result:
    result= result.replace("\t", "")
  elif "\xa0" in result:
    result= result.replace("\xa0", "")
  return result.strip()
