import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

API_KEY = # (your api key here)

client = OpenAI(api_key=API_KEY)

df = pd.read_json("sampled_from_defan_500_NOMATH_3B_temp0.9_topp1.0_maxtokens32.jsonl",lines=True)

SYSTEM = """You are an expert evaluator, judging correctness of a given LLM-generated responses to given questions.

You will be given:
Q: question
R: reference answer (may list multiple required items)
A: model answer

Decide whether A is correct.

Return ONLY valid JSON:
{"label": 0} or {"label": 1} or {"label": null}

Label 0 (correct) if:
- A directly answers Q.
- A matches R in semantic meaning. Synonyms/aliases/paraphrases/different languages etc. are acceptable; order and phrasing do not matter.
- If an answer is expressed in slightly more detail/with slightly more descriptors than expected, that is okay as long as it is factually accurate.
- If R contains multiple required items/entities, A includes ALL of them.

Numeric tolerance:
- Dates and years however must be exact.
- For numeric answers, reasonable approximations are acceptable. E.g., 134,517 approximated as 134,000 or 135,000 is acceptable, 73,891 as 73,000 or 74,000 etc is acceptable.
- PLEASE do not ignore the preceding direction. It is imperative we accept reasonable approximations.

Label 1 (incorrect) if ANY:
- A is empty, a refusal, or not an answer.
- A is missing any required item from R.
- A contradicts Q or R (wrong entity/year/place/value), even if it also mentions a correct token.
- A merely mentions the correct answer string without asserting it as the answer.

Label null 

If uncertain, label 1.
"""

def judge_one(q, r, a):
    a = "" if a is None else str(a).strip()
    if not a:
        return 1

    prompt = f"Q:{q}\nR:{r}\nA:{a}\nReturn JSON:{{\"label\":0|1}}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_completion_tokens=30,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )

    try:
        obj = json.loads(resp.choices[0].message.content)
    except Exception:
        return 1

    return 0 if obj.get("label", 1) == 0 else 1

rows = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    label = judge_one(row["question"], row["answer"], row["model_answer"])
    rows.append({"sample_id": row["sample_id"], "label": label})

labels_df = pd.DataFrame(rows)
labels_df