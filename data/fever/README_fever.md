# Readme FEVER Data

FEVER paper: https://aclanthology.org/N18-1074.pdf

### FEVER dataset files
`shared_task_test_annotations_evidence.jsonl`: annotation of FEVER shared task predictions 

#TODO check where to find the initial shared task testset annotation 

#### Structure of FEVER file:
```
[
    142196,
    {
        "1": 
            {
                "label": "REFUTES", 
                "evidence": [["Descendants_-LRB-2015_film-RRB-", 5, "The film also stars Mitchell Hope , Melanie Paxson , Brenna D'Amico , Sarah Jeffery , Zachary Gibson , Jedidiah Goodacre , Dianne Doan , Dan Payne , Keegan Connor Tracy , Wendy Raquel Robinson , Maz Jobrani , Kathy Najimy , and Kristin Chenoweth . Kristin Chenoweth Kristin Chenoweth Melanie Paxson Melanie Paxson Sarah Jeffery Sarah Jeffery Dan Payne Dan Payne Keegan Connor Tracy Keegan Connor Tracy Wendy Raquel Robinson Wendy Raquel Robinson Maz Jobrani Maz Jobrani Kathy Najimy Kathy Najimy"]], 
                "p_evidence": [], 
                "claim": "Descendants did not star Melanie Paxson."
            }
    }
]
```