# MLingualFC

Welcome to the **MLingualFC** repository! This project focuses on multilingual flowchart translation, natural language description generation, and red teaming / safety evaluations of various LLMs (such as Gemma, Qwen2.5, and Pangea).

## Project Structure

- **Code/**: Contains all the Python scripts used for evaluating different AI models.
  - `flowchart_data_translation.py`: Handles fetching, extracting, and translating `.dot` flowchart models.
  - `*_red_teaming.py`: Scripts for conducting red-teaming evaluations on models (e.g., Gemma, Qwen2.5, Pangea) to assess safety boundaries.
  - `*_plain_text_description.py`: Scripts for plain text description generation related to flowcharts.
  - `*_only_task.py`: Baseline or specialized evaluation scripts corresponding to each language model.
- **Data/**: Contains the flowchart data across different structures (horizontal, tortuous, vertical) and languages:
  - English
  - German
  - Hindi
  - Punjabi
  - Romanian
  - Spanish

## Overview

The purpose of this repository is to analyze how robust and capable modern foundational models are when handling multi-lingual logical flows, and evaluating their responses during targeted red-teaming and prompt-based tasks. The `.dot` graph representation represents complex reasoning structures translated across numerous global languages.

## Getting Started

Make sure you have the required dependencies (such as `deep-translator`, `graphviz`, and model weights/interfaces). You can execute the individual python scripts found in `Code/` to run the respective LLM benchmarks.
