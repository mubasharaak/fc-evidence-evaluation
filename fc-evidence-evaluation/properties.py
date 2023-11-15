import abc
import enum


class Label(enum.Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NEI = "not enough evidence"
    CONFLICTING = "conflicting Evidence/cherrypicking"


class AveritecAnswer(abc.ABC):
    answer: str
    answer_type: str
    source_url: str
    source_medium: str


class AveritecQA(abc.ABC):
    question: str
    answers: list[AveritecAnswer]


class AveritecEntry(abc.ABC):
    claim: str
    required_reannotation: bool
    label: Label
    justification: str
    claim_date: str
    speaker: str
    original_claim_url: str
    fact_checking_article: str
    reporting_source: str
    location_ISO_code: str
    claim_types: list[str]
    fact_checking_strategies: list[str]
    questions: list[AveritecQA]


BASE_PROMPT = """Given a claim and it's associated evidence, decide if the evidence supports the claim, refutes it, 
doesn't give enough information, or gives conflicting information. Explain the decision. Only use the provided 
information and no additional sources or background knowledge.

Examples: 
Claim: All government schools in India are being privatised. 
Evidence: What did India's Union Education 
Minister say about the privatisation of governments schools? New Delhi: There is no plan to privatise primary 
education, the Centre told the Parliament today. This statement was given by Minister of Human Resource Development, 
Ra mesh Pokhriyal Nishank in the Lok Sabha today in response to Kaushalendra Kumar question on whether it is fact 
that NITI Aayog has suggested that Primary Education may be given to the private sector to reduce the burden of 
salary to teachers and other infrastructure. 
Answer: Refute. There is no plan by the Indian government to privatize 
primary education as said by the Minister of Human Resource Development

Claim: South Africans that drink are amongst the top drinkers in the world. 
Evidence: What is the global average 
alcohol consumption in litres of pure alcohol per day? The global averages as of 2016 is 15.1 litres per day. What is 
the daily average of pure alcohol consumption per day in South africa? 29.9 litres. Where does South Africa rank as a 
nation in terms of Daily pure Alcohol consumption? 6th out of 189 countries. 
Answer: support. Is says "amongst the 
top drinkers" not the top, so since they are 6th, this could be plausible

Claim: Anxiety levels among young teenagers dropped during the coronavirus pandemic, a study has suggested 
Evidence: 
What study is this based on? The study was based on a survey of around 1,000 year 9 students in the south west of 
England. It found that students reported lower levels of anxiety when surveyed in April and May of this year compared 
to October 2019. \n\nIt also found an increase in wellbeing but \u201cno large change in risk of depression. Is this 
study backed up by further research? There's nothing wrong with this study in terms of its method or what it claims, 
but it shouldn't necessarily be taken to represent the experience of all children across the country. It can't rule 
out the possibility that the mental toll of the pandemic may have been different across different parts of the 
country, and for people of different ages. \n\nFor example, a study out of the University of Oxford, asked parents 
and carers to report any changes in behavior of their children during a one-month period in lockdown. It found that 
while parents and carers reported that the emotional difficulties of adolescents decreased, the emotional 
difficulties of children aged four to ten increased. \n\nAlso a potential contributor to feelings of anxiety among 
students during the pandemic has been the uncertainty over exam results, which would have primarily affected students 
in years 11 and 13, not year 9 (as the study itself notes). 
Answer: conflicting information. One study shows anxiety 
dropping amongst young teenagers, but another study shows it depends on age and whether child will be receiving exam 
results.

Claim: There is a global average for the number of judges and magistrates to number of people in Kenya. 
Evidence: How 
many magistrates were their in Kenya in 2020? No answer could be found. Is there a global average for the number of 
judges compared to population? No answer could be found. What is the population of Kenya? 47.6 million 
Answer: not 
enough information. The answers do not support or refute the claim as there is no evidence to look at in reference to 
the claim.

Claim: {} 
Evidence: {}
Answer: {} 
"""


COT_PROMPT = """Given a claim and it's associated evidence, decide and give an explanation if the evidence (1) 
supports the claim, (2) refutes it, (3) does not give enough information, or gives (4) conflicting information. Think 
step-by-step and provide the reasoning steps. Only use the provided information and no additional sources or 
background knowledge.

Claim: South Africans that drink are amongst the top drinkers in the world. 
Evidence: The global average alcohol 
consumption in litres of pure alcohol per day is 15.1 litres. The daily average of pure alcohol consumption per day 
in South africa is 29.9 litres. South Africa ranks as a nation in terms of Daily pure Alcohol consumption 6th out of 
189 countries. 
Answer: 
1. Understand the Claim:
The claim asserts that "South Africans that drink are amongst the top drinkers in the world."
2. Break Down the Evidence:
Three central points from the evidence:
The global daily average for alcohol consumption is 15.1 litres of pure alcohol.
The daily average of pure alcohol consumption in South Africa stands at 29.9 litres.
On a global scale, considering daily pure alcohol consumption, South Africa is ranked 6th out of 189 countries.
3. Evaluate the Claim in Light of the Evidence:
The claim posits that South Africans are among the top global consumers of alcohol.
Evidence shows South Africans consume alcohol at a rate almost double the global average (29.9 litres compared to 15.1 
litres). The ranking of South Africa as 6th out of 189 countries accentuates the idea that they are "amongst the top" 
alcohol consumers worldwide.
4. Draw a Conclusion:The evidence unequivocally supports the claim that South Africans that drink are amongst the top 
drinkers in the world.

Claim: {} 
Evidence: {}
Answer: {} 
"""

TOT_PROMPT = """Imagine three different experts are answering this question. All experts will write down 1 step of 
their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert 
realises they're wrong at any point then they leave.

Given a claim and it's associated evidence, the task is to decide and explain if the evidence (1) supports the claim, 
(2) refutes it, (3) does not give enough information, or gives (4) conflicting information. The experts should only 
use the provided information and no additional sources or background knowledge. Finally, summarize the output and 
explain why the experts arrive to a certain conclusion in few sentences.

Claim: South Africans that drink are amongst the top drinkers in the world. 
Evidence: The global average alcohol 
consumption in litres of pure alcohol per day is 15.1 litres. The daily average of pure alcohol consumption per day 
in South africa is 29.9 litres. South Africa ranks as a nation in terms of Daily pure Alcohol consumption 6th out of 
189 countries. 
Answer: Expert A:
Step 1: I'll start by understanding the claim thoroughly. It is suggesting a comparative position of South Africans 
in global alcohol consumption. The crucial piece of evidence would be South Africa's rank or a metric that shows its 
position relative to other countries.

Expert B:
Step 1: First, I will focus on the consumption values mentioned. The evidence points out a daily average for South 
Africa. I will assess this value in the context of the global average to determine its relative position.

Expert C:
Step 1: "Amongst the top" in the claim implies a ranking. The evidence must provide clarity on South Africa's 
standing compared to other countries for alcohol consumption.

Expert A:
Step 2: The evidence clearly indicates that South Africa is positioned 6th out of 189 countries in terms of daily 
alcohol consumption. This ranking places South Africa in the top echelons of alcohol-consuming nations.

Expert B:
Step 2: The daily average alcohol consumption in South Africa is nearly double the global average (29.9 litres 
compared to 15.1 litres). This provides a clear indication that South Africans consume more alcohol than the average 
global citizen.

Expert C:
Step 2: Not only does South Africa have a significantly higher daily average consumption than the global average, 
but it's also ranked 6th out of 189 countries. This data solidifies South Africa's position among the top drinkers 
globally.

Expert A:
Step 3: Given the rank and the higher than average consumption values, the evidence supports the claim that South 
Africans who drink are among the top drinkers in the world.

Expert B:
Step 3: Evaluating both the consumption values and South Africa's rank, the evidence undoubtedly supports the claim.

Expert C:
Step 3: The combined insights from the ranking and the daily consumption values confirm that the evidence supports 
the claim.

Summary: All three experts concur that the evidence provided supports the claim that South Africans that drink are 
amongst the top drinkers in the world. Their conclusion is based on the significantly higher daily average alcohol 
consumption in South Africa compared to the global average, as well as South Africa's high ranking (6th out of 189 
countries) in global alcohol consumption.

Claim: {} 
Evidence: {}
Answer: {} 
"""