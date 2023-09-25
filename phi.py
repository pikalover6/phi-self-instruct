import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("totally-not-an-llm/karkadann-1.3b", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("totally-not-an-llm/karkadann-1.3b", trust_remote_code=True, torch_dtype=torch.bfloat16)

def generate(inp, max_length=2048, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.2):
    inputs = tokenizer(inp, return_tensors="pt", return_attention_mask=False).to("cuda")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=do_sample, temperature=temperature, top_p=top_p, use_cache=True, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.batch_decode(outputs)[0]
    return text

def chat(inp, max_length=2048, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.2, prompt_format="### Instruction:\n{}\n### Response:"):
    text = generate(prompt_format.format(inp), max_length, do_sample, temperature, top_p, repetition_penalty)
    return text

mars = """
Mars is the fourth planet and the furthest terrestrial planet from the Sun. The reddish color of its surface is due to finely grained iron(III) oxide dust in the soil, giving it the nickname "the Red Planet".[21][22] Mars's radius is second smallest among the planets in the Solar System at 3,389.5 km (2,106 mi). The Martian dichotomy is visible on the surface: on average, the terrain on Mars's northern hemisphere is flatterand lower than its southern hemisphere. Mars has a thin atmosphere made primarily of carbon dioxide and two irregularly shaped natural satellites: Phobos and Deimos.

Geologically, Mars is fairly active, with dust devils sweeping across the landscape and marsquakes (Martian analog to earthquakes) trembling underneath the ground. The surface of Mars hosts a large shield volcano (Olympus Mons) and one of the largest canyons in the Solar System (Valles Marineris). Mars's significant orbital eccentricity and axial tilt cause large seasonal changes to the polar ice caps' coverage and temperature swings between −110 °C (−166 °F) to 35 °C (95 °F) on the surface. A Martian solar day (sol) is equal to 24.5 hours and a Martian solar year is equal to 1.88 Earth years.

Like the other planets in the Solar System, Mars was formed approximately 4.5 billion years ago. During the Noachian period from about 4.1 to 3.7 billion years ago, Mars's surface was marked by meteor impacts, valley formation, erosion, and the possible presence of water oceans. The Hesperian period from 3.7 to 3.2–2 billion years ago was dominated by widespread volcanic activity and flooding that carved immense outflow channels. The Amazonian period, which continues to the present, was marked by the wind's influence on geological processes. It is unknown whether life has ever existed on Mars.

Mars is among the brightest objects in Earth's sky, and thus has been known from ancient times. Its high-contrast albedo features make it a common subject for viewing with a telescope. Since the late 20th century,Mars has been explored by uncrewed spacecraft and rovers, with the first flyby by the Mariner 4 probe in 1965, the first Mars orbiter by the Mars 2 probe in 1971, and the first landing by Viking 1 in 1976. As of 2023, there are at least 11 active probes orbiting Mars or at the Martian surface. Currently, Mars is an attractive target for the first future interplanetary human missions.
"""

txt = generate("The following is a college-level textbook with detailed and high level information.\Title: The planets of the solar system.\nChapter 1 (Mars):\n"+ mars +"\nChapter 2 (Earth):\n")

print(txt)
