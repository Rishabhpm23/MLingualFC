# # Cell 2: Imports and language configuration

import os
import re
import shutil
import subprocess
import textwrap
from deep_translator import GoogleTranslator
from IPython.display import Image, display

# Input folder containing the original English .dot files
SOURCE_FOLDER = ""

# Output base folder.
OUTPUT_BASE = ""

LANG_CONFIG = {
    "german": {
        "code": "de",
        "font": "Noto Sans", 
        "wrap_width": 20  
    },
    "romanian": {
        "code": "ro",
        "font": "Noto Sans", 
        "wrap_width": 20
    },
    "hindi": {
        "code": "hi",
        "font": "Noto Sans Devanagari", 
        "wrap_width": 20
    },
    "punjabi": {
        "code": "pa",
        "font": "Noto Sans Gurmukhi",
        "wrap_width": 20
    }
}

label_pattern = re.compile(r'label\s*=\s*"(.*?)"', re.DOTALL)

RECREATE_OUTPUT = True


# # Cell 3: Helper functions

TRANSLATION_CACHE = {}

def safe_translate(text, translator, lang_code):
    """
    Translate text safely and cache results.
    """
    try:
        clean_text = (
            text.replace("\\n", " ")
                .replace("\n", " ")
                .strip()
        )
        clean_text = re.sub(r"\s+", " ", clean_text)

        if not clean_text:
            return text

        cache_key = (lang_code, clean_text)
        if cache_key in TRANSLATION_CACHE:
            return TRANSLATION_CACHE[cache_key]

        translated = translator.translate(clean_text)
        TRANSLATION_CACHE[cache_key] = translated
        return translated

    except Exception:
        return text

def wrap_graphviz_text(text, width=18):
    """
    Wrap translated text into multiple lines so Graphviz renders a narrower node.
    """
    if not text:
        return text

    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return text

    wrapped_lines = []

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            wrapped_lines.append("")
            continue

        lines = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=False,
            break_on_hyphens=False
        )

        if not lines:
            lines = [paragraph]
        elif len(lines) == 1 and len(lines[0]) > width:
            lines = textwrap.wrap(
                paragraph,
                width=width,
                break_long_words=True,
                break_on_hyphens=False
            )

        wrapped_lines.extend(lines)

    return "\n".join(wrapped_lines)

def dot_escape(text):
    """
    Escape text for Graphviz DOT labels.
    """
    return (
        text.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", r"\n")
    )

def translate_dot_content(content, translator, lang_code, wrap_width=18):
    """
    Translate every label in a DOT file, wrap the translated text,
    and keep it safe for Graphviz.
    """
    labels = label_pattern.findall(content)
    unique_labels = list(dict.fromkeys(labels))

    translated_map = {}
    for label in unique_labels:
        translated = safe_translate(label, translator, lang_code)
        wrapped = wrap_graphviz_text(translated, width=wrap_width)
        translated_map[label] = dot_escape(wrapped)

    def replace_label(match):
        original_label = match.group(1)
        translated_label = translated_map.get(original_label, dot_escape(original_label))
        return f'label="{translated_label}"'

    return label_pattern.sub(replace_label, content)

def add_font_settings(dot_content, font_name):
    """
    Insert Graphviz font settings.
    """
    font_block = f'''
graph [fontname="{font_name}", nodesep="0.35", ranksep="0.5", pad="0.15"];
node [fontname="{font_name}", margin="0.10,0.08"];
edge [fontname="{font_name}"];
'''
    if 'graph [fontname=' not in dot_content:
        dot_content = dot_content.replace("{", "{\n" + font_block, 1)
    return dot_content


for language_name in LANG_CONFIG.keys():
    output_folder = f"{OUTPUT_BASE}_{language_name}"

    if RECREATE_OUTPUT and os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

print("Output folders are ready.")


# # Cell 4: Translate the DOT files into all five languages

for language_name, cfg in LANG_CONFIG.items():
    lang_code = cfg["code"]
    wrap_width = cfg["wrap_width"]
    output_folder = f"{OUTPUT_BASE}_{language_name}"

    translator = GoogleTranslator(source="en", target=lang_code)

    for root, dirs, files in os.walk(SOURCE_FOLDER):
        relative_path = os.path.relpath(root, SOURCE_FOLDER)
        output_root = os.path.join(output_folder, relative_path)
        os.makedirs(output_root, exist_ok=True)

        for file in files:
            if not file.endswith(".dot"):
                continue

            input_path = os.path.join(root, file)

            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()

            translated_content = translate_dot_content(
                content,
                translator,
                lang_code=lang_code,
                wrap_width=wrap_width
            )

            base_name = os.path.splitext(file)[0]
            output_file = f"{base_name}_{language_name}.dot"
            output_path = os.path.join(output_root, output_file)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(translated_content)

            print(f"Translated: {output_path}")

    print(f"\nDone translating to {language_name.upper()}\n")


# # Cell 5: Add font settings for each language

for language_name, cfg in LANG_CONFIG.items():
    folder_path = f"{OUTPUT_BASE}_{language_name}"
    font_name = cfg["font"]

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(f"_{language_name}.dot"):
                path = os.path.join(root, file)

                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                updated_content = add_font_settings(content, font_name)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(updated_content)

    print(f"Font settings added for {language_name.upper()}")


# # Cell 6: Render all translated DOT files to PNG

for language_name in LANG_CONFIG.keys():
    folder_path = f"{OUTPUT_BASE}_{language_name}"

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(f"_{language_name}.dot"):
                dot_path = os.path.join(root, file)
                png_path = os.path.splitext(dot_path)[0] + ".png"

                subprocess.run(
                    ["dot", "-Gdpi=300", "-Tpng", dot_path, "-o", png_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

    print(f"{language_name.upper()} flowcharts generated correctly.")


# # Cell 7: Verifying Dataset Structure

for language_name in LANG_CONFIG.keys():
    folder_path = f"{OUTPUT_BASE}_{language_name}"
    dot_count = 0
    png_count = 0
    unexpected_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(f"_{language_name}.dot"):
                dot_count += 1
            elif file.endswith(f"_{language_name}.png"):
                png_count += 1
            else:
                unexpected_files.append(path)

    print(f"\n{language_name.upper()}:")
    print(f"DOT files: {dot_count}")
    print(f"PNG files: {png_count}")

    if unexpected_files:
        print("Unexpected files found:")
        for path in unexpected_files[:10]:
            print(path)


# # Cell 8: quick visual check on one sample file

import random

all_original_dots = []
for root, dirs, files in os.walk(SOURCE_FOLDER):
    for file in files:
        if file.endswith(".dot"):
            all_original_dots.append(os.path.join(root, file))

sample_original = random.choice(all_original_dots)
relative_path = os.path.relpath(sample_original, SOURCE_FOLDER)

print("Selected original file:")
print(sample_original)
print()

with open(sample_original, "r", encoding="utf-8") as f:
    print("--- ORIGINAL DOT ---")
    print(f.read())

for language_name in LANG_CONFIG.keys():
    translated_dot = os.path.join(
        f"{OUTPUT_BASE}_{language_name}",
        relative_path
    ).replace(".dot", f"_{language_name}.dot")

    translated_png = translated_dot.replace(".dot", ".png")

    print(f"\n--- {language_name.upper()} DOT ---")
    print(translated_dot)

    if os.path.exists(translated_dot):
        with open(translated_dot, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("Translated DOT not found.")

    if os.path.exists(translated_png):
        print(f"\n--- {language_name.upper()} PNG ---")
        display(Image(filename=translated_png))
    else:
        print("Translated PNG not found.")

