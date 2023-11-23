# Readme Averitec Data

Averitec paper: https://arxiv.org/pdf/2305.13117.pdf

### Averitec dataset 

`averitec_train.json`: Averitec training set

`averitec_test.json`: Averitec test set

`averitec_dev.json`: Averitec dev set

#### Structure of Averitec dataset files

```commandline
{
        "claim": "In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial.",
        "required_reannotation": false,
        "label": "Refuted",
        "justification": "The answer and sources show that the claim was published in a fake news site so the claim is refuted.",
        "claim_date": "31-10-2020",
        "speaker": null,
        "original_claim_url": null,
        "fact_checking_article": "https://web.archive.org/web/20201130144023/https://checkyourfact.com/2020/11/03/fact-check-sean-connery-letter-steve-jobs-apple-1998/",
        "reporting_source": "Facebook",
        "location_ISO_code": null,
        "claim_types": [
            "Event/Property Claim"
        ],
        "fact_checking_strategies": [
            "Written Evidence"
        ],
        "questions": [
            {
                "question": "Where was the claim first published",
                "answers": [
                    {
                        "answer": "It was first published on Sccopertino",
                        "answer_type": "Abstractive",
                        "source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/",
                        "source_medium": "Web text"
                    }
                ]
            },
            {
                "question": "What kind of website is Scoopertino",
                "answers": [
                    {
                        "answer": "Scoopertino is an imaginary news organization devoted to ferreting out the most relevant stories in the world of Apple, whether or not they actually occurred - says their about page",
                        "answer_type": "Extractive",
                        "source_url": "https://web.archive.org/web/20201202085933/https://scoopertino.com/about-scoopertino/",
                        "source_medium": "Web text"
                    }
                ]
            }
        ]
    }

```