"""
moderate.py
"""
import pandas as pd

from moderators.aws import main as aws_main
from moderators.azure import main as azure_main
from moderators.gpt_oss import main as gptoss_main
from moderators.hf_models import main as hf_main
from moderators.llamaguard3 import main as llamaguard_main
from moderators.modelarmor import main as modelarmor_main
from moderators.openai_moderation import main as openai_main
from moderators.perspective import main as perspective_main
from moderators.qwen3guard import main as qwen3guard_main


def main():
    """
    Run all moderators on the RabakBench dataset and save the results.
    """
    
    for lang in ['en', 'ms', 'ta', 'zh']: 
        print(f"Running moderators for {lang}...")
        df = pd.read_csv(f"data/{lang}/rabakbench_{lang}.csv")
        
        # Closed source models
        aws_main(df, lang)
        azure_main(df, lang)
        modelarmor_main(df, lang)
        openai_main(df, lang)
        perspective_main(df, lang)
        
        # Open source models
        llamaguard_main(df, lang)
        qwen3guard_main(df, lang)
        gptoss_main(df, lang)
        hf_main(df, model_name='llamaguard4', lang=lang, batch_size=8)
        hf_main(df, model_name='wildguard', lang=lang, batch_size=32)
        hf_main(df, model_name='shieldgemma', lang=lang, batch_size=8)
        hf_main(df, model_name='granite-guardian', lang=lang, batch_size=8)
        hf_main(df, model_name='duoguard', lang=lang, batch_size=64)
        hf_main(df, model_name='polyguard', lang=lang, batch_size=64)
    

if __name__ == "__main__":
    main()
