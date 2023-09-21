import phi
import json

categories = {
    "general_knowledge":
    {
        "nature": {
            "prompt": "Generate {} college-level test questions related to nature in a numbered list, WITHOUT giving the answers"
        },
        "world history": {
            "prompt": "Generate {} college-level test questions related to world history in a numbered list, WITHOUT giving the answers"
        }
    },
    "coding":
    {
        "general": {
            "prompt": "Generate {} python-related programming test questions in a numbered list, WITHOUT giving the answers"
        }
    },
    "reasoning":
    {
        "general": {
            "prompt": "Generate {} mathematical reasoning questions in a numbered list, WITHOUT giving the answers"
        }
    }
}

seeds = []

for cat in categories:
    for subcat in categories[cat]:
        seed = categories[cat][subcat]["prompt"].format("10")
        seeds += [a.split(" ", 1)[1] for a in phi.chat(seed, temperature=0.4).replace("<|endoftext|>", "").split("\n")[1:]]

with open("seed_prompts.json", "w") as outfile:
    json.dump(seeds, outfile)
