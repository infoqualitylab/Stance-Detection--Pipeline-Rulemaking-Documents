
"""
This code converts the JSON file into an CSV by pulling relevant information and cleaning as needed. 
It also saves the CSV file locally in a format that is ideal for annotation tasks.
@author: Suchita Hothur
"""
import json
import pandas as pd
import re


#Function to extract organisation data from json
def extract_organisations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    elements = data.get('elements', [])
    results = []

    i = 0
    while i < len(elements):
        el = elements[i]
        text = el.get('Text', '')

        # Detect organisation heading — italic containing "Organization:"
        is_org_heading = (
            'Organization:' in text and
            el.get('Font', {}).get('italic', False)
        )

        if is_org_heading:
            org_name = text.replace('Organization:', '').strip()
            org_texts = []

            # Collect following paragraph elements until the next org heading or major heading
            j = i + 1
            while j < len(elements):
                next_el = elements[j]
                next_text = next_el.get('Text', '')
                next_path = next_el.get('Path', '')

                # Stop if we hit another organisation heading
                if 'Organization:' in next_text and next_el.get('Font', {}).get('italic', False):
                    break

                # Stop if we hit a top-level section heading (Title or H1)
                if next_path in ('//Document/Title', '//Document/H1') or next_path.startswith('//Document/Title'):
                    break
                # Skip footnote elements by path
                if 'Footnote' in next_path:
                    j += 1
                    continue
                # Include paragraph and span content, skip sub-headings
                if any(p in next_path for p in ['//Document/P', '//Document/ParagraphSpan']):
                    # Skip lines that begin with a footnote number (e.g. '29 U.S. ...' or '30 Reflects...')
                    first_word = next_text.strip().split()[0] if next_text.strip() else ''
                    if not first_word.isdigit():
                        org_texts.append(next_text.strip())

                j += 1

            results.append({
                'organisation': org_name,
                'text': '\n\n'.join(org_texts)
            })

            i = j  # jump ahead
        else:
            i += 1

    return results

#function to save csv as a file
def save_to_csv(results, output_path='CSV/shortened2.csv'):
    import csv
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['organisation', 'text'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} record(s) to {output_path}")


def clean_csv(input_path, output_path, text_column= 'text'):
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found.")
   
    #Regex formula to clean text of EPA references 
    df[text_column] = df[text_column].apply(
        lambda text: re.sub(r'\[EPA[^\]]+\]', '', text).strip() if isinstance(text, str) else text
    )
    #Regex to remove the number references to footnotes within text. An example looks like : ".35 The EPA..."
    df[text_column] = df[text_column].apply(
        lambda text: re.sub(r'\.\d+', '.', text).strip() if isinstance(text, str) else text
    )
  
    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")


if __name__ == '__main__':
    import sys

    json_path =  'JSON/shortened_sec13_2.json'
    csv_path =  'CSV/shortened2.csv'

    results = extract_organisations(json_path)

    if not results:
        print("No organisations found.")
    else:
        save_to_csv(results, csv_path)
        clean_csv (csv_path,csv_path)
        






