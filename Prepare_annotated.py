"""
This code prepares the annotated text (in JSON format) to be fed into a predicitve model by converting it into CSV. 
The resulting dataframe is saved locally.
@author: ADOBE pvt Ltd
Modified by: Suchita Hothur
"""
import json
import pandas as pd

LABEL_TO_STANCE = {
    "Feasibility of producing vehicles":
        "It is feasible to produce Electric Vehicles or EVs at the scale of predicted demand",
    "Consumer desires":
        "Consumers want to buy more electric vehicles (EVs)",
    "Market challenges":
        "The market is ready to support increased consumption of electric vehicles (EVs)",
    "Vehicle utility":
        "Electric vehicles (EVs) have equal or more utility when compared to "
        "Internal Combustion Engine (ICE) vehicles",
    "Cost of fuel/ charging ":
        "Charging electric vehicles (EVs) will be cost effective",
    "Charging infrastructure":
        "There will be enough charging infrastructure to meet new electric vehicle (EV) adoption rates",
    "Cost of purchasing vehicle":
        "Electric vehicles (EVs) are equally or more economical than "
        "Internal Combustion Engine vehicles",
}

#Function to extract data from JSON and save to csv
def extract_from_JSON(input_path,output_path):
    # Create the empty DataFrame
    columns = ['organisation', 'text', 'label', 'stance_pred']
    df = pd.DataFrame(columns=columns)
    with open(input_path, 'r') as f:
        data = json.load(f)
    
        rows = []
        # Loop through each org
        for entry in data:
            organisation = entry['data']['organisation']
            main_data = entry['annotations'][0]['result']
            no_snip=len(main_data)
            #Looping through each snippet
            i=0
            while (i<no_snip-1):
                entry1=main_data[i]
                entry2=main_data[i+1]
                if(entry1['value']['start']==entry2['value']['start']):
                    df.loc[len(df)]=[organisation, entry1['value']['text'],entry1['value']['labels'][0],entry2['value']['labels'][0]]
                    i=i+2
                else:
                    i=i+1
    
    df.to_csv(output_path, index=False)


def recode_csv(path):
    df = pd.read_csv(path)
    df = df[df["label"].str.strip() != "Other"].reset_index(drop=True)

    # Recode label -> label_stance
    df["label_stance"] = df["label"].str.strip().map(
        {k.strip(): v for k, v in LABEL_TO_STANCE.items()}
    )
    df.to_csv(path, index=False)


if __name__ == '__main__':
    input_path="Annotation/Annotation_2026-03-18.json"
    output_path="LM_models/model_ready.csv"

    #extract_from_JSON(input_path,output_path)
    recode_csv(output_path)


  