import os
import numpy as np
import pickle as pkl
import random
from openai import OpenAI
from numpy.linalg import norm
from sklearn.metrics import f1_score

API_KEY = ""
baseURLs = {
    "qwen": "http://127.0.0.1:8000/v1", 
    "deepseek": "http://127.0.0.1:8001/v1"
}

modelName = {
    "qwen": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
}

indicator_entity = {
    "positive": "respiratory specimens testing positive for influenza",
    # TODO
}

window_size = 20
time_step = "week"
domain = "healthcare"
target_event_description = "It will rain in the next 24 hours" # TODO
label_block = "rain or not rain" # TODO 

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def contextualize_whole(
    indicator: str, 
    model: str,
    system_template_path = f"../prompt_templates/contextualize_whole_system.txt",
    user_template_path = f"../prompt_templates/contextualize_whole_user.txt",
):
    # Load dataset
    with open(f'time_series_{indicator}.pkl', 'rb') as f:
        data = pkl.load(f)
    with open(f'indices_{indicator}.pkl', 'rb') as f:
        indices = pkl.load(f)
    data_size = data.shape[0]

    # Load prompt templates from txt files
    system_template = _load_text(system_template_path).strip()
    user_template = _load_text(user_template_path).strip()

    # Fill system prompt
    entity = indicator_entity[indicator]
    system_prompt = system_template.format(DOMAIN = domain)
    
    # Run 
    client = OpenAI(api_key=API_KEY, base_url = baseURLs[model])
    os.makedirs(f"{model}_summary", exist_ok=True)
    os.makedirs(f"{model}_summary/whole", exist_ok=True)
    
    for i in indices[:10]: # TODO: remove :10
        output_path = f"{model}_summary/whole/{indicator}_{i}.txt"
        if os.path.exists(output_path):
            print(f"Skipping {output_path} (already exists)")
            continue
        
        data_window = data[i:i+window_size]
        
        if indicator == 'positive':
            total_specimens = '|'.join([x for x in data_window[:,1]])
            total_a = '|'.join([x for x in data_window[:,2]])
            total_b = '|'.join([x for x in data_window[:,3]])
            pos_rate = '|'.join([str(f'{float(x):.2f}') for x in data_window[:,4]])
            a_rate = '|'.join([str(f'{float(x):.2f}') for x in data_window[:,5]])
            b_rate = '|'.join([str(f'{float(x):.2f}') for x in data_window[:,6]])

            # Build INDICATOR_BLOCK
            indicator_block = (
                f"- Number of specimens tested: {total_specimens}\n"
                f"- Number of positive specimens for Influenza A: {total_a}\n"
                f"- Number of positive specimens for Influenza B: {total_b}\n"
                f"- Ratio of positive specimens (%): {pos_rate}\n"
                f"- Ratio of positive specimens for Influenza A (%): {a_rate}\n"
                f"- Ratio of positive specimens for Influenza B (%): {b_rate}"
            )


        # Fill user prompt
        user_prompt = user_template.format(
            ENTITY=entity,
            WINDOW_SIZE=window_size,
            TIME_STEP=time_step,
            INDICATOR_BLOCK=indicator_block,
        )

        response = client.chat.completions.create(
            model=modelName[model],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
        )
        text = response.choices[0].message.content
        with open(output_path, 'w') as f:
            f.write(text)
        print(f"generation: {output_path}")

def contextualize_channel(
    indicator: str, 
    model: str,
    system_template_path = f"../prompt_templates/contextualize_channel_system.txt",
    user_template_path = f"../prompt_templates/contextualize_channel_user.txt",
):
    # Load dataset
    with open(f'indices_{indicator}.pkl', 'rb') as f:
        indices = pkl.load(f)
    with open(f'time_series_{indicator}.pkl', 'rb') as f:
        data = pkl.load(f)
    data_size = data.shape[0]

    # Load prompt templates from txt files
    system_template = _load_text(system_template_path).strip()
    user_template = _load_text(user_template_path).strip()

    # Fill system prompt
    entity = indicator_entity[indicator]
    system_prompt = system_template.format(DOMAIN = domain)
    channels = [
        ("Number of specimens tested", 1, "count"),
        ("Number of positive specimens for Influenza A", 2, "count"),
        ("Number of positive specimens for Influenza B", 3, "count"),
        ("Ratio of positive specimens (%)", 4, "rate"),
        ("Ratio of positive specimens for Influenza A (%)", 5, "rate"),
        ("Ratio of positive specimens for Influenza B (%)", 6, "rate"),
    ] # TODO: for another indicator, it has different channels !!
    
    # Run 
    client = OpenAI(api_key=API_KEY, base_url = baseURLs[model])
    os.makedirs(f"{model}_summary", exist_ok=True)
    os.makedirs(f"{model}_summary/channel", exist_ok=True)
    
    for i in indices[:10]: # TODO: remove :10
        
        output_path = f"{model}_summary/channel/{indicator}_{i}.txt"
        if os.path.exists(output_path):
            print(f"Skipping {output_path} (already exists)")
            continue
        
        data_window = data[i:i+window_size]

        combined_summaries = []

        for indicator_name, col_idx, kind in channels:
            if kind == "rate":
                indicator_series = "|".join([f"{float(x):.2f}" for x in data_window[:, col_idx]])
            else:
                indicator_series = "|".join([str(x) for x in data_window[:, col_idx]])

            user_prompt = user_template.format(
                ENTITY=entity,
                WINDOW_SIZE=window_size,
                TIME_STEP=time_step,
                INDICATOR_NAME=indicator_name,
                INDICATOR_SERIES=indicator_series,
            )

            response = client.chat.completions.create(
                model=modelName[model],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=512,  # by-channel outputs are short; saves cost/time
                top_p=1,
            )
            text = response.choices[0].message.content
            combined_summaries.append(f"- {indicator_name}: {text}")

        final_text = "\n".join(combined_summaries)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"generation: {output_path}")

def predict(
    city: str, 
    model: str, 
    k: int=5,
    summary_type = "whole",
    system_template_path: str = "../prompt_templates/predict_system.txt",
    user_template_path: str = "../prompt_templates/predict_user.txt",
):
    # Load Dataset
    with open('indices.pkl', 'rb') as f:
        indices = pkl.load(f)
    with open('dates.pkl', 'rb') as f:
        dates = pkl.load(f)
    with open(f'time_series_{city}.pkl', 'rb') as f:
        data = pkl.load(f)
    with open(f'rain_{city}.pkl', 'rb') as f:
        rains = pkl.load(f)
    texts = {}
    for i in indices:
        with open(os.path.join(f'{model}_summary/{summary_type}', f'{city}_{i}.txt'), 'r') as f:
            text = f.read()
            texts[i] = text
    # Split
    data_size = len(indices)
    num_train = int(data_size * 0.6)
    num_test = int(data_size * 0.2)
    num_vali = data_size - num_train - num_test

    seq_len_day = 1
    idx_train = np.arange(num_train - seq_len_day)
    idx_valid = np.arange(num_train - seq_len_day, num_train + num_vali - seq_len_day)
    idx_test = np.arange(num_train + num_vali - seq_len_day, num_train + num_vali + num_test - seq_len_day)

    # Load embeddings for retrieval
    with open(f'../../encoder/embeddings/weather_{city}.pkl', 'rb') as f:
        embs = pkl.load(f)

    text_emb = {}
    for _i, i in enumerate(indices[:-1]):
        text_emb[i] = embs[_i]

    def cos(a, b):
        cos_sim = np.dot(a, b)/(norm(a)*norm(b))
        return cos_sim
    
    # Load prompt templates
    system_template = _load_text(system_template_path).strip()
    user_template = _load_text(user_template_path).strip()

    # Fill system prompt from template. 
    entity = city_full_name[city]
    system_prompt = system_template.format(
        DOMAIN = domain,
        TARGET_EVENT_DESCRIPTION = target_event_description,
    )

    # Run
    client = OpenAI(api_key=API_KEY, base_url = baseURLs[model])
    random.seed(2024)

    os.makedirs(f"{model}_predict_summary-{summary_type}_k-{k}", exist_ok=True)

    for _i in idx_test:
        i = indices[_i]

        output_path = f"{model}_predict_summary-{summary_type}_k-{k}/{city}_{i}_ref.txt"

        if os.path.exists(output_path):
            print(f"Skipping {output_path} (already exists)")
            continue
        
        # Retrieval: pick k nearest train examples by cosine similarity (descending)
        sim = [-cos(text_emb[i], text_emb[indices[ii]]) for ii in idx_train]
        _j_list = np.argsort(sim)
        
        # Build EXAMPLE_BLOCK for template
        example_lines = []
        for ex_idx in range(k):
            _j = _j_list[ex_idx]
            j = indices[_j]

            example_lines.append(f"Example #{ex_idx+1}:")
            example_lines.append(f"Summary: {texts[j]}")

            outcome_is_rain = bool(rains[_j + 1])
            outcome_text = "rain" if outcome_is_rain else "not rain"
            example_lines.append(f"Outcome: {outcome_text}")
            example_lines.append("")

        example_block = "\n".join(example_lines).strip()
        
        # Fill user prompt from template
        user_prompt = user_template.format(
            EXAMPLE_BLOCK=example_block,
            CURRENT_SUMMARY=texts[i],
            LABEL_BLOCK=label_block,
            # Optional placeholders if your template includes them later:
            ENTITY=entity,
            WINDOW_SIZE=window_size,
            TIME_STEP=time_step,
            K=k,
        )
        
        response = client.chat.completions.create(
            model=modelName[model],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=256,  # prediction is short; saves cost/time
            top_p=1,
        )
        text = response.choices[0].message.content
        with open(output_path, 'w') as f:
            f.write(f'{text}')
        print(f"prediction: {output_path}")

def main():
    for indicator in ["positive"]: #  TODO: another indicator
        for model in ["qwen", "deepseek"]:
            contextualize_whole(indicator, model)
            contextualize_channel(indicator, model)
            # predict(city, model)

if __name__ == "__main__":
    main()