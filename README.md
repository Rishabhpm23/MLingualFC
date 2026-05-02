# MLingualFC

Welcome to the **MLingualFC** repository! This project focuses on multilingual flowchart translation, natural language description generation, and red teaming / safety evaluations of various VLMs (such as Gemma4, Qwen2.5, and Pangea).

## Project Structure

- **Code/**: Contains all the Python scripts used for evaluating different AI models.
  - `flowchart_data_translation.py`: Handles fetching, extracting, and translating `.dot` flowchart models.
  - `*_red_teaming.py`: Scripts for conducting red-teaming evaluations on models (e.g., Gemma, Qwen2.5, Pangea) to assess safety boundaries.
  - `*_plain_text_description.py`: Scripts for plain text description generation related to flowcharts.
  - `*_only_task.py`: Baseline or specialized evaluation scripts corresponding to each language model.
