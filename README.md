
<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to contribute to an end-to-end pipeline that extracts, cleans and formats data from policy related documents to predict the opinion of third-part stakeholders to proposed federal regulations. It will be used to help understand the complex stances held by organisations towards regulations. We will be able to analyse how alignments amongst blocs of these organisations look and ultimately how this amalgam of opinions affects thr rule making process. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* Adobe PDF API
* PyTorch
* HuggingFace Transformers
* Scikit-learn
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running clone this git repository locally

### Prerequisites
Softwares:
   1. ADOBE PDF extract API
   2. Label Studio (Annotation)

Python:
   1. Python version 3 or greater
   2. json
   3. pandas
   4. Transformers from HuggingFace
   5. PyTorch


## Project Structure
 
```
.
├── scripts/Extract_comments_from_PDF.py              # Converts Data from pdf to a structured JSON 
├── scripts/Convert_JSON_to_CSV.py                    # Extracts relevant data and cleans from the JSON to produce CSV
├── scripts/Prepare_annotated.py                      # Converts annotated JSOn data back into model ready CSV, while adding
├── LM_models/Longformer.py                           # Stance detection with Longformer and evaluation
├── LM_models/RoBERTA.py                              # Stance detection with RoBerta and evaluation
├── LM_models/modelready.csv                          # Fully prepared dataset with annotations
├── LM_models/stance_results_RoBERTA.csv              # Results from RoBERTa     
├── LM_models/stance_results_longformer.csv             
├── sec13_2--pages2121-2135.json                      # Initial jason from extraction
└── README.md
```
 
---

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- USAGE EXAMPLES -->

## Usage

- Extract_comments_from_PDF.py:
         
         This script takes the raw pdf as input and outputs a structured JSON containing all contents of the PDF.
         
         Run this script by entering "python3 Extract_comments_from_PDF.py." The input pdf is provided within the main code. Change this to provide different inputs. The default input is PDFs/Section13_comments_shortened2.pdf 
         
         To run this file you will need to install the Adobe PDF extract API. [Details for installation can be found here] (https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/)

- Convert_JSON_to_CSV.py
         
         This script takes the structured JSON from the previous step and extracts two key elements: the organisation name and the main comment. It also cleans comment text by removing line breaks, in text references, references to external documents etc. The out put is a CSV file with two columns: "organization" and "text".
         
         To run this file install pandas and json. Run teh command "python3 Convert_JSON_to_CSV.py." Input and output paths can be specified within code itself. 

- Prepare_annotated.py
         
         This script utilises the JSON file exported from the manual annotation software to create model-ready CSV data. It extracts organisation name, topic label, stance label and comment text. Each row corresponds to a snippet of a comment made by an organisation in relation to a certain topic label. 
         
         The script also drops rows with topic label "Other" and converts labels to targeted stances. For example Charging infrastructure" label gets mapped to "There will be enough charging infrastructure to meet new electric vehicle (EV) adoption rates."

         Edit input and output paths in the main body of the code and run this script with "python3 Prepare_annotated.py."
- LM_models/Longformer.py and LM_models/RoBERTA.py 
         
         These scripts replicate popular LLM models using Hugging face's Transformers library. I used the base RoBERTA model for natural language inference and base Longformer model.

         Run this line to conduct stance detection:
            python3  LM_models/RoBERTA.py --input LM_models/model_ready.csv
            python3 LM_models/Longformer.py --input LM_models/model_ready.csv


Note: This project benefited from LLM usage. Claude sonnet 4.6 was used to fix technical issues and generate code snippets.
<!-- CONTACT -->
## Contact

Main contributor: Suchita Hothur - shothur04@gmail.com
Advisors: Heng Zheng - hengzheng@uky.edu and Jodi Schneider - jodi@wisc.edu

Project Link: [Githib Link Goes Here](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>






