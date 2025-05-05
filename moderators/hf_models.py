import gc

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from moderators.format_utils import (
    build_wildguard_prompts, 
    parse_wildguard_outputs, 
    build_shieldgemma_prompts, 
    parse_shieldgemma_outputs,
    build_granite_guardian_prompts,
    parse_granite_guardian_outputs,
    build_duoguard_prompts,
    parse_duoguard_outputs
)

MODELS = {
    "wildguard": "allenai/wildguard", #7b
    "shieldgemma": "google/shieldgemma-9b", #9b
    "granite-guardian": "ibm-granite/granite-guardian-3.2-5b", #5b
    "duoguard": "DuoGuard/DuoGuard-0.5B" #0.5B
}


class HFClassifier:
    def __init__(self, model_name: str, batch_size: int = 16, device: str = "cuda"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        # Special handling for DuoGuard which is a sequence classification model
        if model_name == "duoguard":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODELS[model_name],
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODELS[model_name],
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], use_fast=False)
        
    def classify(self, items: list[dict[str, str]]) -> list[dict]:
        results = []
        for batch_start in tqdm(
            range(0, len(items), self.batch_size), 
            total=len(items) // self.batch_size,
            desc="Classifying batches"
        ):
            batch = items[batch_start : batch_start + self.batch_size]
            results.extend(self.classify_batch(batch))

        return results
 
 
    def classify_batch(self, batch: list[dict[str, str]]) -> list[dict]:
        assert self.model is not None

        if self.model_name == "duoguard":
            return self._classify_batch_duoguard(batch)
        else:
            formatted_prompts = self.build_prompts(batch)
            tokenized_inputs = self.tokenizer(
                formatted_prompts,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            generated_outputs = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            batch_outputs = self.tokenizer.batch_decode(
                generated_outputs.sequences, skip_special_tokens=True
            )
            batch_logits = torch.stack(generated_outputs.scores, dim=1)
            
            outputs = self.parse_outputs(_, batch_logits)
    
            return outputs

    def _classify_batch_duoguard(self, batch: list[str]) -> list[dict]:
        # DuoGuard specific implementation. Will only run when model_name = "duoguard".
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # DuoGuard outputs a 12-dimensional vector (one probability per subcategory).
            logits = outputs.logits  # shape: (batch_size, 12)

        # Input batch is passed into parse_output function as placeholder
        outputs = self.parse_outputs(batch, logits)
            
        return outputs

    def build_prompts(self, batch: list[str]) -> list[str]:
        if self.model_name == "wildguard":
            return build_wildguard_prompts(batch)
        elif self.model_name == "shieldgemma":
            return build_shieldgemma_prompts(batch)
        elif self.model_name == "granite-guardian":
            return build_granite_guardian_prompts(batch, self.tokenizer)
        elif self.model_name == "duoguard":
            return build_duoguard_prompts(batch)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        
    def parse_outputs(self, batch: list[str], logits: torch.Tensor) -> list[dict]:
        if self.model_name == "wildguard":
            return parse_wildguard_outputs(batch)
        elif self.model_name == "shieldgemma":
            return parse_shieldgemma_outputs(batch, logits, self.tokenizer)
        elif self.model_name == "granite-guardian":
            return parse_granite_guardian_outputs(batch)
        elif self.model_name == "duoguard":
            return parse_duoguard_outputs(logits)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
    
    
def main(df, model_name: str, lang: str = 'en', batch_size: int = 16):
    # Prepare messages in batches 
    df = df[["prompt_id","text"]].copy()
    messages = df["text"].tolist()
    
    classifier = HFClassifier(model_name=model_name, batch_size=batch_size)
    outputs = classifier.classify(messages)
    
    categories = outputs[0].keys()
    for category in categories:
        df[category] = [result[category] for result in outputs]
        
    df.to_csv(f"data/{lang}/rabakbench_{lang}_{model_name}.csv", index=False)

    # Clear gpu 
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == "__main__":
    main()