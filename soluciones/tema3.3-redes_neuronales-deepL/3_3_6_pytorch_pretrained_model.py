# Ejemplos de modelos preentrenados: 
# • PyTorch Transformers

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""Goldfinger is the seventh novel in Ian Fleming's James Bond series.
First published in 1959, it centres on Bond's investigation into the
gold-smuggling activities of Auric Goldfinger, who is suspected of being
connected to Soviet counter-intelligence. Bond uncovers Goldfinger's plot
involving the gold reserves at Fort Knox (pictured). In Goldfinger, Fleming
presents the character of James Bond as a more complex individual than in the
previous novels. A theme of Bond as a St George figure is echoed by the fact
that Bond is a British Secret Service agent sorting out an American problem.
Fleming probably based the gold-obsessed character of Goldfinger on the American
gold tycoon Charles W. Engelhard Jr. On its release, the novel went to the top
of the best-seller lists. It was adapted as the third James Bond feature film of
the Eon Productions series, released in 1964 and starring Sean Connery as Bond.
"""

questions = [
    "What is the year of the first post of Goldfinger?",
]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")