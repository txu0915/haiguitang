"""Main program to launch the online Haiguitang app"""
import os
import logging
import gradio
import yaml
from sentence_transformers import CrossEncoder
from utils import relevance_probablistic_score, pop_question_from_source


problems_dir = 'data'
problems_fp = 'problems.yaml'
model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

logging.info(f"Reading problems to solve from local...")
with open(os.path.join(problems_dir, problems_fp), 'r') as f:
    problems = yaml.safe_load(f)

q_a_mapping = {puzzle: problems['Answers'][i] for i, puzzle in enumerate(problems['Puzzles'])}

logging.info(f"Problems list formed successfully, in total {len(problems['Puzzles'])} have been loaded...")

model = CrossEncoder(model_name, max_length=512)
logging.info(f"Model named {model_name} has been successfully loaded...")

initial_question, initial_answer = pop_question_from_source(q_a_mapping)

first_launch, time_to_show_question = True, True


def predict(message, history):
    global first_launch, initial_question, initial_answer, time_to_show_question

    if not history and not first_launch:
        curr_question, curr_answer = pop_question_from_source(q_a_mapping)
        initial_question, initial_answer = curr_question, curr_answer
        time_to_show_question = True
    else:
        curr_question, curr_answer = initial_question, initial_answer

    first_launch = False

    relevancy = relevance_probablistic_score(curr_answer, message, model)
    print(message, curr_answer, relevancy)

    if time_to_show_question:
        response = f"Please take a look at this puzzle and think of a solution: \n{curr_question}"
        time_to_show_question = False
    else:
        if 0.0 < relevancy <= 0.33:
            response = f"Your score is only {relevancy: .3f}, please think about it..."
        elif 0.33 < relevancy <= 0.67:
            response = f"You have gotten most of it, score {relevancy: .3f}, hurry up!"
        else:
            response = f"You got it, score {relevancy: .3f}"

    yield response

gradio.ChatInterface(predict).queue().launch(share=True)