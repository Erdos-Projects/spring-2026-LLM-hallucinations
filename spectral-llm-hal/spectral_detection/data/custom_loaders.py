
SAMPLE1 = {
    "question": "A pita is a type of what?", 
    "choices" : [ "Fresh Fruit", "flat bread", "French tart", "fried bean dip" ],
    "reference_answer": "B", 
    "dataset": "mmlu",
    "topic_label": ""
}

SAMPLE2 = {
    "question": "Prosthetic groups are:", 
    "choices" : [ "required by all enzymes in the cell.","loosely bound to enzymes via hydrogen bonds.",
                   "sites on the enzyme molecule that permit allosteric modification of enzyme activity.",
                   "tightly bound to enzymes and are required for their activity."
                ],
    "reference_answer": "D", 
    "dataset": "mmlu",
    "topic_label": ""
}
    
SAMPLE3 = {
    "question": "Select the best English interpretation of the given proposition, using the following translation key: Ax: x is an apartment Hx: x is a house Lx: x is large Bxy: x is bigger than y (∃x)[Hx • (∀y)(Ay ⊃ Bxy)]", 
    "choices" : ["Some houses are smaller than all apartments.", 
                 "Every house is bigger than every apartment.", 
                 "Some apartment is smaller than every house.", 
                 "Some houses are bigger than every apartment."],
    "reference_answer": "D", 
    "dataset": "mmlu",
    "topic_label": ""
}

SAMPLE4 = {
    "question": "Which of these is a slang term for 'police'?", 
    "choices" : ["fuzz",
                 "shrinks",
                 "bean counters",
                 "aardvarks"
                ],
    "reference_answer" : "A",
    "dataset" : "mmlu",
    "topic_label" : ""
}

SAMPLE5 = {
    "question": "Most of the radiation in Earth’s biosphere is", 
    "choices" : ["natural background radiation",
                 "the result of military activities",
                 "from nuclear power plants",
                 "in the form of cosmic rays"
                ],
    "reference_answer" : "A",
    "dataset" : "mmlu",
    "topic_label" : ""
}

def load_mmlu_special():
    return [
        SAMPLE1, 
        SAMPLE2, 
        SAMPLE3
        ]


def load_mmlu_special_2():
    return [
        SAMPLE4, 
        SAMPLE5, 
        SAMPLE3
        ]