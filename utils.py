"""Helper utils functions to build the app"""
import numpy
import gradio
from sentence_transformers import CrossEncoder


def logit2prob(logit: float) -> float:
    """
    compute the survival probability given a numerical logit value.
    :param logit: float format of logit value
    :return: float format of computed survival probability
    """
    odds = numpy.exp(logit)
    prob = odds / (1 + odds)
    return prob


def relevance_probablistic_score(textA: str, textB: str,
                                 model: CrossEncoder) -> float:
    """
    Given two texts, compute the relevancy score in probablistic manner
    :param textA: context A
    :param textB: context B
    :return: probablistic relevancy score of contexts
    """
    score = model.predict([[textA, textB]])[0]
    return logit2prob(score)


def pop_question_from_source(q_a_mapping: dict) -> tuple[str, str]:
    """
    switch to a new question when clicked
    :param q_a_mapping:
    :return: display a new prompt text on screen as a new question is generated, input text model to be fed into LLMs
    """
    randomIdx = numpy.random.choice(list(range(len(q_a_mapping.items()))))
    puzzle, answer = list(q_a_mapping.items())[randomIdx]
    return puzzle, answer


def upfront_question(q_a_mapping: dict) -> tuple[str, str]:
    """
    display the upfront question, no clicking needed
    :param q_a_mapping:
    :return: display a new prompt text on screen as a new question is generated, input text model to be fed into LLMs
    """
    randomIdx = numpy.random.choice(list(range(len(q_a_mapping.items()))))
    puzzle, answer = list(q_a_mapping.items())[randomIdx]
    return puzzle, answer


def form_display_and_inputs(q_a_mapping: dict) -> tuple[gradio.templates.TextArea, gradio.Textbox]:
    """
    given a puzzle & question list, compose the display and inputs
    :return:
    """
    puzzle, answer = pop_question_from_source(q_a_mapping)
    display_text = gradio.TextArea(f'Please take a look at the following question: {answer}')
    input_text = gradio.Textbox(label=puzzle, lines=6)
    return display_text, input_text


def prompt(answer: str) -> str:
    """
    given answer form a prompt
    :param answer:
    :return:
    """
    return f'Please take a look at the following question: {answer}'


