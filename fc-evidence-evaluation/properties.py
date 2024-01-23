import enum
from dataclasses import dataclass
from typing import List
from typing import Optional, Union

from aenum import MultiValueEnum


class Dataset(enum.Enum):
    FEVER = "fever"
    FEVER_REANNOTATION = "fever_reannotation"
    AVERITEC = "averitec"
    AVERITEC_AFTER_P4 = "averitec_after_p4"
    HOVER = "hover"
    VITAMINC = "vitaminc"


class PromptTypes(enum.Enum):
    BASE = "base"
    COT = "cot"
    TOT = "tot"
    ATOMIC_FACTS = "atomic"
    SCORE = "score"


class Label(MultiValueEnum):
    REFUTED = "refuted", "refutes", "refute", 0, "0", "contradiction", "c", "not_supported", "contradict"
    SUPPORTED = "supported", "supports", "support", 1, "1", "entailment", "e", "entail"
    NEI = "not enough evidence", 2, "2", "neutral", "n", "conflicting evidence/cherrypicking", "not enough info", "not enough information", "nei"


LABEL_DICT = {
    Label.SUPPORTED: 0,
    Label.NEI: 1,
    Label.REFUTED: 2,
}

LABEL_DICT_REVERSE = {
    0: "support",
    1: "not enough information",
    2: "refute",
}


@dataclass
class OpenAIResponse:
    claim: str
    evidence: str
    response: str
    gold: str


@dataclass
class AveritecAnswer:
    answer: str
    answer_type: str
    boolean_explanation: Optional[str]

    # def __init__(self, answer, answer_type, boolean_explanation=None):
    #     self.answer = answer
    #     self.answer_type = answer_type
    #     self.boolean_explanation = boolean_explanation


@dataclass
class AveritecQA:
    question: str
    answers: List[AveritecAnswer]

    # def __init__(self, question: str, answers: list[AveritecAnswer]):
    #     self.question = question
    #     self.answers = answers


@dataclass
class AveritecEntry:
    claim: str
    label: str
    justification: str
    evidence: Union[List[AveritecQA], str]

    # def __init__(self, claim: str, label: str, justification: str, evidence: list[AveritecQA]):
    #     self.claim = claim
    #     self.label = label
    #     self.justification = justification
    #     self.evidence = evidence


FEVER_DATASET_PATH = "shared_task_test_annotations_evidence.jsonl"
AVERITEC_TRAIN_FILENAME = "averitec_train.json"
AVERITEC_TEST_FILENAME = "averitec_test.json"
AVERITEC_DEV_FILENAME = "averitec_dev.json"

AVERITEC_INIT_FILES = [AVERITEC_TRAIN_FILENAME, AVERITEC_TEST_FILENAME, AVERITEC_DEV_FILENAME]

BASE_PROMPT = """Given a claim and it's associated evidence, decide if the evidence supports the claim, refutes it, 
or doesn't give enough information. Only use the provided information and no additional sources or background knowledge.

------
Examples: 
Claim: All government schools in India are being privatised. 
Evidence: What did India's Union Education 
Minister say about the privatisation of governments schools? New Delhi: There is no plan to privatise primary 
education, the Centre told the Parliament today. This statement was given by Minister of Human Resource Development, 
Ra mesh Pokhriyal Nishank in the Lok Sabha today in response to Kaushalendra Kumar question on whether it is fact 
that NITI Aayog has suggested that Primary Education may be given to the private sector to reduce the burden of 
salary to teachers and other infrastructure. 
Answer: refute.

Claim: South Africans that drink are amongst the top drinkers in the world. 
Evidence: What is the global average 
alcohol consumption in litres of pure alcohol per day? The global averages as of 2016 is 15.1 litres per day. What is 
the daily average of pure alcohol consumption per day in South africa? 29.9 litres. Where does South Africa rank as a 
nation in terms of Daily pure Alcohol consumption? 6th out of 189 countries. 
Answer: support

Claim: There is a global average for the number of judges and magistrates to number of people in Kenya. 
Evidence: How 
many magistrates were their in Kenya in 2020? No answer could be found. Is there a global average for the number of 
judges compared to population? No answer could be found. What is the population of Kenya? 47.6 million 
Answer: not enough information
------
The answer should be a json with a key named label
Claim: {} 
Evidence: {}
Answer:
"""

COT_PROMPT = """Given a claim and it's associated evidence, decide if the evidence supports the claim, refutes it, 
or doesn't give enough information. Explain the reasoning step-by-step before giving the answer. Only use the provided 
information and no additional sources or background knowledge.
-----
Examples: 
Claim: South Africans that drink are amongst the top drinkers in the world. 
Evidence: What is the global average 
alcohol consumption in litres of pure alcohol per day? The global averages as of 2016 is 15.1 litres per day. What is 
the daily average of pure alcohol consumption per day in South africa? 29.9 litres. Where does South Africa rank as a 
nation in terms of Daily pure Alcohol consumption? 6th out of 189 countries. 
Answer: Explanation: The claim stays "amongst the top drinkers" not the top first, so since they are 6th, this could be plausible. The answer is support. Label: support.

Claim: All government schools in India are being privatised. 
Evidence: What did India's Union Education Minister say about the privatisation of governments schools? New Delhi: There is no plan to privatise primary 
education, the Centre told the Parliament today. This statement was given by Minister of Human Resource Development, 
Ra mesh Pokhriyal Nishank in the Lok Sabha today in response to Kaushalendra Kumar question on whether it is fact 
that NITI Aayog has suggested that Primary Education may be given to the private sector to reduce the burden of 
salary to teachers and other infrastructure. 
Answer: Explanation: There is no plan by the Indian government to privatize primary education as said by the Minister of Human Resource Development. The claim is clearly refuted and therefore the answer is refute. Label: refute.

Claim: There is a global average for the number of judges and magistrates to number of people in Kenya. 
Evidence: How 
many magistrates were their in Kenya in 2020? No answer could be found. Is there a global average for the number of 
judges compared to population? No answer could be found. What is the population of Kenya? 47.6 million 
Answer: Explanation: The evidence does neither support nor refute the claim that a global average for the number of judges and magistrates to number of people exists in Kenya. The answer is not enough information. Label: not enough information

Claim: An IndyCar race driver drove a Formula 1 car designed by Peter McCool during the 2007 Formula One season.
Evidence: The Super Aguri SA07 was Super Aguri F1's Formula One car for the 2007 Formula One season. It was designed by Peter McCool and was driven by Takuma Sato and Anthony Davidson. Takuma Sato (佐藤 琢磨, Satō Takuma, born 28 January 1977), nicknamed "Taku", is a Japanese professional racing driver. He competes part-time in the IndyCar Series, driving the No. 11 Honda for Chip Ganassi Racing.
Answer: Explanation: Takuma Sato is an IndyCar race driver who drove the Super Aguri SA07 for the 2007 Formula One season. The evidence states that this car was designed by Peter McCool. Hence the answer is support.Label: support 

Claim: Rhythm Nation was incapable of being performed on Britain's Got Talent.
Evidence: It has been covered by Pink , Crystal Kay , and Girls ' Generation and has also been performed on Glee , The X-Factor , and Britain 's Got Talent .
Answer: Explanation: The Evidence states that the song Rhythm Nation was performed on Britain's Got Talent and therefore clearly refutes the claim that it was incapable to be performed. The answer is refute. Label: refute. 

Claim: Alloy media platforms have a monthly reach of less than 100 million unique visitors .
Evidence: According to comScore , Alloy media platforms reach over 95 million unique visitors each month , including over half of the age 12-34 internet users .
Answer: Explanation: While the evidence mentions that the platform reaches over 95 million unique visitors per month, it does not state clearly if the number is lower than 100 million. Hence the evidence does not contain enough information to decide if the claim is supported or refuted. Label: not enough information.

Claim: The American film, television and theater actress who was a star in the film the Matchmaker and also received the 40th AFI Life Achievement Award was born April 24, 1934.
Evidence: The Matchmaker is a 1958 American comedy film directed by Joseph Anthony. The film stars Shirley Booth in her final film, Anthony Perkins, and Shirley MacLaine. Shirley MacLaine (born Shirley MacLean Beaty; April 24, 1934)[1] is an American film, television and theater actress and author. Known for her portrayals of quirky, strong-willed and eccentric women, she has received numerous accolades over her eight-decade career, including an Academy Award, an Emmy Award, two BAFTA Awards, six Golden Globe Awards, two Volpi Cups, two Silver Bears, and the 40th AFI Life Achievement Award. 
Answer: Explanation: The actress mentioned in the claim is Shirley Booth as she was starred in the Matchmaker, received the 40th AFI Life Achievement Award and was born on April 24, 1934 as stated in the evidence. The answer is support. Label: support.

Claim: Bambi, is based on a book by the American author Felix Salten, The Country Bears is not.
Evidence: Bambi, a Life in the Woods (German title: Bambi: Eine Lebensgeschichte aus dem Walde) is a 1923 Austrian coming-of-age novel written by Felix Salten, and originally published in Berlin by Ullstein Verlag. The Country Bears is a 2002 American musical road comedy[2] film directed by Peter Hastings, produced by Walt Disney Pictures, and based on the Disney theme park attraction Country Bear Jamboree.
Answer: Explanation: The evidence supports that Bambi is a book by Felix Slaten and The Country Bears is not. However, it mentions that Felix Salten is American while the evidence tells us he was Austrian. The claims is refuted. Label: refute.

Claim: The film Deliver Us from Evil , released in 2014 , has Eric Bana , Edgar Ramírez , Sean Harris , Olivia Munn , and Joel McHale as the main stars .
Evidence: Despite mixed to negative reviews by critics , the film, which was released in 2014, was a box office success , grossing $ 87.9 million against a $ 30 million budget .
Answer: Explanation: While the evidence confirms that the movie was released in 2014 it misses information about the actors starred in the movie. Hence the answer is not enough information given. Label: not enough information.

-----
The answer should be a json with two keys: explanation, label.

Claim: {} 
Evidence: {}
Answer:
"""

NEI_EXAMPLE = """
Claim: There is a global average for the number of judges and magistrates to number of people in Kenya. 
Evidence: How 
many magistrates were their in Kenya in 2020? No answer could be found. Is there a global average for the number of 
judges compared to population? No answer could be found. What is the population of Kenya? 47.6 million 
Answer: The evidence does not support or refute the claim as there is no evidence to look at in reference to 
the claim. The answer is not enough information. Not enough information
"""

TOT_PROMPT = """Imagine three different experts are answering this question. All experts will write down 1 step of 
their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert 
realises they're wrong at any point then they leave.

Given a claim and it's associated evidence, the task is to decide and explain if the evidence (1) supports the claim, 
(2) refutes it, (3) does not give enough information, or gives (4) conflicting information. The experts should only 
use the provided information and no additional sources or background knowledge. Finally, summarize the output and 
explain why the experts arrive to a certain conclusion in few sentences.
-----
Example:

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
countries) in global alcohol consumption. support.
-----
The answer should be a json with two keys: summary, label.

Claim: {} 
Evidence: {}
Answer:
"""

PROMPT_SHOTS = """Claim: After losing the defamation case held against them by Jay Shah, the son of the India Home 
Affairs Minister Amit Shah, Indian news website, The Wire, has apologised to the ministers son for holding false 
allegations against him. 
Evidence: Did The Wire lose a defamation case brought by Jay Shah? No answer could be found. Did The Wire offer 
an apology over the defamation case brought by Jay Shah? Probably not. In court in 2018 The Wire\u2019s counsel Nitya 
Ramakrishnan \"was emphatic that there was no question of an apology\". 
Answer: not enough information. There lacks evidence as to whether the Wire lost the defamation case, and if there was
issued an apology between 2018 and the time of the claim.

Claim: mRNA vaccines increase the risk of death for COVID-19 patients, provoke cytokine storms, and turn patients 
into genetically modified organisms. 
Evidence: Does the mRNA vaccines increase the risk of system over-reactions for 
COVID-19 patients? No. Brewer said one of the problems mRNA vaccine research had to solve was the tendency of the 
messenger material - which triggers cells to make proteins that trigger the body's immune reaction -- to break down 
and not last long enough in the body to be effective. \"It's been the opposite problem,\" to the one Cahill 
described, Brewer said. \"The mRNA would degrade too quickly or it would not get into the cells.\" New approaches 
have packaged the messenger RNA so that it can trigger the body to develop defenses against the SARS-CoV-2 virus that 
causes COVID. Does mRNA vaccines  genetically modify organisms within COVID-19 patients? No. One of the advantages of 
mRNA vaccines is they do not integrate at all and they don't even get into the nucleus of the cell. All the action 
happens in the cytoplasm so there's no concern about any genetic modification. -  Prof. Timothy Brewer, a UCLA David 
Geffen School of Medicine infectious disease researcher. Does the mRNA vaccine increase cytokine storms? Yes. 
SARS-CoV-2 can activate monocytes/macrophages, dendritic cells, T cells, mast cells, neutrophils, and induce cytokine 
storm in the lung. How can the mRNA Vaccine increase death? Anaphylactic reactions can occur with any vaccine, 
but are usually extremely rare\u2014about one per 1 million doses. As of 19 December, the United States had seen six 
cases of anaphylaxis among 272,001 people who received the COVID-19 vaccine, according to a recent presentation by 
Thomas Clark of the U.S. Centers for Disease Control and Prevention (CDC). 
Answer: conflicting information. There are some aspects which are supported such as the vaccine provoking a cytokine 
storm but the other parts are refuted so the claim has conflicting evidence.

Claim: Vanguard is a shareholder of BlackRock... which, incidentally, is a major shareholder of Microsoft The 
property of Bill Gates, who owns Pfizer (the company that sells the miracle vaccine) and is currently the first 
sponsor of WHO. 
Evidence: Is Vangaurd a shareholder in Microsoft? Yes. Vangaurd is the largest shareholder in 
Microsoft. Is Microsoft Bill Gates' property? No. Bill Gates is stepping down from the board of Microsoft, 
the software giant he helped found more than four decades ago.\n\nHowever, Mr. Gates remains one of the largest 
individual shareholders of Microsoft. As of December, he held more than 100 million shares of Microsoft, roughly 1.3 
percent of the company\u2019s shares. His stock is worth about $16 billion. He said he would remain active at the 
company and would work closely with Satya Nadella, Microsoft\u2019s chief executive. Is Vangaurd a shareholder in 
Pfizer?. Yes, Vangaurd, Pfizer's largest institutional shareholder, ranks first on the list of Pfizer's largest 
institutional shareholders.Is BlackRock a shareholder in Pfizer? Yes. BlackRock is the second largest shareholder of 
Pfizer. Is BlackRock a shareholder in Microsoft? Yes. BlackRock is the second largest shareholder in Microsoft. 
Answer: cherrypicking. Although the Q&A pairs are missing some points made in the claim, the points they do make are 
cherry picked, to form the claim overall\nThe 2nd Q&A pair does state that bill gates stepped down but it fails to 
point out he is still a majority shareholder/owner

Claim: Sardar Vallabhbhai Patel airport in Ahmedabad has been renamed Adani Airports. 
Evidence: Why was Sardar 
Vallabhbhai Patel airport renamed Adani airport? The airport was not renamed: Adani Group has taken over the 
operations of Sardar Vallabhbhai Patel International Airport in Ahmedabad. The Airports Authority of India completed 
all the procedure and handed over the airport operations to the Adani Group. The ceremony of handing over the key of 
the airport was held at the midnight on November 06. The management, operations and development of the airport will 
now be handled by the Adani Group from November 07 for the period of 50 years. Sardar Vallabhbhai Patel International 
Airport recorded footfall of 1.14 crore passengers in 2019-20. This step by Adani Group is expected to strengthen and 
expand the airport business. 
Answer: Refute. It has not been renamed, the Adani group took over operations.

Claim: On 9th December 2020, 50,000 Chinese soldiers were bombed and killed in Maine and the event listed as an 
earthquake. 
Evidence: Was there an earthquake in Maine on December 9, 2020? Yes. A minor earthquake was detected in 
Maine on December 9th, 2020. Were 50,000 Chinese soldiers killed in Maine on December 9, 2020? No. There is nothing 
to substantiate that 50,000 Chinese soldiers landed in Maine. Was a bomb dropped on Maine on December 9, 2020? "There 
is no evidence of a bomb being dropped,\" Ken Clark, chief of Calais Fire and Emerging Medical Services in Maine. 
Answer: Refute. No evidence to support.

Claim: Biden administration temporarily freezes arms sales to Saudi Arabia and the United Arab Emirates. 
Evidence: 
What did President Biden do to the arms deals between the US and the United Arab emirates? President Joe Biden is 
temporarily freezing weapons sales to Saudi Arabia and the United Arab Emirates, pledging to review a set of 
controversial arms deals former President Donald Trump struck with the two U.S. allies in the waning days of his 
administration. 
Answer: Support. From the Q&As you can see that Biden did in fact temporarily freeze arms sales. So the claim is
 supported.

Claim: Russia launched a cyberattack against the U.S. government. 
Evidence: Who was responsible for this attack? Many 
expert analysts and researchers firmly believe a Russian group known as the Cozy Bear or SVR can be accredited for an 
attack against the United States government. In what is known as a SolarWinds attack. What happened during this 
attack? this Russian group is believed to have launched this attack by leveraging a loophole within the SolarWinds 
platform. SolarWinds is a network performance monitoring tools used by hundreds of organizations to manage the 
availability, performance, and reporting of their network systems. Does this pose a grave danger to the US? Yes. 
Analysis into all of the affected processes, organizations, and systems is still ongoing and developing. As such, 
many organizations around the U.S. that use the SolarWinds products have been analyzing their networks and logs to 
validate whether they\u2019ve been affected. This includes several government agencies, schools, hospitals, 
and companies. One of the organizations adversely impacted that poses a grave danger to the welfare of the U.S. is 
the Treasury and Commerce Department.\n\nUnfortunately, with security being so new for many organizations, 
many lack the ability to properly assess whether they\u2019ve been impacted by the attack. This will therefore reduce 
the ability to properly notify all Americans that may have had their information compromised. Answer: not enough 
information. There was a suspected cyber attack but there's no concrete evidence to point towards Russia.

Claim: Denzel Washington said, ''I support police over BLM, don't put them down!'' 
Evidence: Did Denzel Washington 
say, \"I support police over BLM, don't put them down! No. Denzel Washington said in a Jan. 27, 2021 interview with 
Yahoo Entertainment, \"I have the utmost respect for what they do, for what our soldiers do, [people] that sacrifice 
their lives,\" Washington tells Yahoo Entertainment during a recent interview (watch above). \"I just don\u2019t care 
for people who put those kind of people down. If it weren\u2019t for them, we would not have the freedom to complain 
about what they do. Has Denzel Washington spoken on the Black Lives Matter movement? Yes. At the premiere of his new 
film, \"Fences.\" Denzel Washington discussed the Black Lives Matter movement. \n\nThe interviewer that year wondered 
whether Washington thought the Black Lives Matter movement had helped race relations in the United States. 
\n\nWashington skirted the question and said, \"Listen. We live in America, and in America we have the freedom to 
express ourselves. We shouldn\u2019t take that for granted. So whatever the movements are, whether you agree with 
them or don\u2019t, they have the right to express themselves. So that\u2019s one of the great things about being in 
this country, that you do have the right to protest. 
Answer:  Refute. Mr Washington's words were taken out of context as it is evident from the 1st answer. Therefore the 
label is refuted

"""

ATOMIC_PROMPT = """
You will get as input a claim and evidence. 
Please verify the correctness of the claim following the following steps.
1. Break down the claim in independent facts. Each fact should be a separate sentence. 
2. Only break down the claim into facts, not the evidence!
3. Evaluate each fact individually using the given evidence only. Do not use additional sources or background knowledge.
4. Finally summarise how many facts are (1.) supported by the evidence, (2.) contradict with the evidence, (3.) are not verifiable given the evidence. Therefore, generate a dictionary with three keys: supports, contradicts, not enough information.

-----
Examples:

Claim: Mukesh Ambani, richest man in Asia had surgery for pancreatic cancer at Sloan Kettering, New York, US cancer speciality hospital on October 30, 2020.
Evidence: When was the photograph taken of Mukesh Ambani on the Facebook post claiming he had been diagnosed with pancreatic cancer and had undergone surgery? The photograph was taken on September 5, 2020. When was a video filmed of  Mukesh Ambani at the virtual launch of NK Singh's book Portrait of Power? The video was filmed on October 19, 2020. What date was the  Facebook post which confirmed Mukesh Ambani had lost 30 kgs, been diagnosed with pancreatic cancer and had had liver transplant surgery? The Facebook post was dated November 2, 2020. Where was Mukesh's photo of him supposedly recieving surgery actually taken? It was taken by Manushree Vijayvergiya who shared her experience of meeting Mukesh and Isha Ambani in a cafe in Liechtenstein.
Facts: - Mukesh Ambani is the richest man in Asia.
- Mukesh Ambani had surgery for pancreatic cancer.
- The surgery took place at Sloan Kettering, a cancer specialty hospital in New York, US.
- The surgery occurred on October 30, 2020.
Fact check: - Mukesh Ambani is the richest man in Asia. Not enough information given.
- Mukesh Ambani had surgery for pancreatic cancer. Not enough information given.
- The surgery took place at Sloan Kettering, a cancer specialty hospital in New York, US. Not enough information given.
- The surgery occurred on October 30, 2020. The evidence shows other appearances by Ambani shortly before and after October 30, 2020. This conflicts with the fact “The surgery occurred on October 30, 2020”.
Output: {"support": 0, "refute": 0, "not enough info": 3}


Output: 0 facts have the label support, 1 fact has the label contradicts, 3 facts have the label not enough information

Claim: Millions of jobs in the US were lost during Donald Trump's US presidency.
Evidence: How many people were in employment in 2017? 145,627,000 people as of January 2017. How many people were in employment in 2020? 141,735,000 people in September 2020. How many people in employment did the economy lose under Trump's presidency? The economy lost an estimate of 3,892,000 people in employment.
Facts: - Donald Trump was US president. 
 - Millions of jobs in the US were lost during his US presidency.
Fact check: 
- Donald Trump was US president. Supported, the evidence mentions Trump’s presidency indicating that he was president of the US.
 - Millions of jobs in the US were lost during his US presidency. The evidence supports this statement.
Output: 2 facts have the label support, 0 facts has the label contradicts, 0 facts have the label not enough information

-----
The answer should be a json with three keys: support, refute, not enough information. The label should be the number of facts falling into this category.

Claim: {}
Evidence: {}
Fact check:
Output:
"""

SCORE_PROMPT = """
Score the following claim given evidence on a continual scale from 0 (worst) to 100 (best).
A score of 0 means “The evidence is completely unrelated to the claim.”
A score of 50 means “Approximately half of the claim is supported by the evidence. For the remaining part not enough information is given in the evidence.”
A score of 100 means “The claim is fully supported by the evidence text” or “Evidence clearly contradicts something stated in the claim.”

-----
Examples:

Claim: Mukesh Ambani, richest man in Asia had surgery for pancreatic cancer at Sloan Kettering, New York, US cancer speciality hospital on October 30, 2020.
Evidence: When was the photograph taken of Mukesh Ambani on the Facebook post claiming he had been diagnosed with pancreatic cancer and had undergone surgery? The photograph was taken on September 5, 2020. When was a video filmed of  Mukesh Ambani at the virtual launch of NK Singh's book Portrait of Power? The video was filmed on October 19, 2020. What date was the  Facebook post which confirmed Mukesh Ambani had lost 30 kgs, been diagnosed with pancreatic cancer and had had liver transplant surgery? The Facebook post was dated November 2, 2020. Where was Mukesh's photo of him supposedly recieving surgery actually taken? It was taken by Manushree Vijayvergiya who shared her experience of meeting Mukesh and Isha Ambani in a cafe in Liechtenstein.
Score: 100. The evidence clearly contradicts with the surgery occurred on “October 30, 2020”. The evidence shows other appearances by Ambani shortly before and after October 30, 2020. 

Claim: Millions of jobs in the US were lost during Donald Trump's US presidency.
Evidence: How many people were in employment in 2017? 145,627,000 people as of January 2017. How many people were in employment in 2020? 141,735,000 people in September 2020. How many people in employment did the economy lose under Trump's presidency? The economy lost an estimate of 3,892,000 people in employment.
Score: 100. The evidence mentions Trump’s presidency indicating that he was president of the US. The evidence supports that millions of jobs in the US were lost during his US presidency.

Claim: In 1963, Collins became one of the third group of astronauts selected by NASA and he  served as the back-up Command Module Pilot for the Gemini 7 mission. 
Evidence: What was the profession of Collins? Collins was an astronaut at NASA?
Was he one of the astronauts selected for the third group by NASA? Yes. Collins became one of the third group of astronauts selected by NASA.
When did NASA select a third group of astronauts? The third group of astronauts was selected by NASA in 1963.
Score: 75. While the evidence supports three facts mentioned in the claim (collins was astronaut, selected by NASA for third group, selection was in 1963), it doesn’t mention that he was part of the Gemini 7 mission as back-up Command Module Pilot. 

Claim: The government of the Solomon Islands has decided to ban social media platform Facebook.
Evidence: Who is  the judge who had a trial regarding Epstein and the Deutsche Bank, whose son has been killed now and her husband has been shot? Deutsche Bank also connects Epstein to Judge Esther Salas, a federal judge whose son was killed and her husband injured in a shooting at their New Jersey home in July. \n\nSalas has presided over a class action lawsuit brought against the bank by investors who claim it made false and misleading statements about its anti-money laundering policies, according to the Associated Press, and because it failed to monitor \"high-risk\" customers like Epstein.\n\nThe lawyer suspected in the shooting at Salas\u2019 home was Roy Den Hollander, a self-described \"anti-feminist\" who had compiled a dossier on her and her family and arrived at their home that morning carrying a FedEx package.
Score: 0. The evidence is complete unrelated to the claim. The evidence neither refutes nor supports the claim.

-----

Claim: {}
Evidence: {}
Score:
"""

REFERENCE_PROMPT = """
TODO compare Averitec prediction to Averitec annotated evidence for reference based evaluation.
"""

PROMPT_MAPPING = {
    PromptTypes.BASE: BASE_PROMPT,  # is the PSEUDO prompt
    PromptTypes.COT: COT_PROMPT,  # is the PSEUDO prompt that uses COT prompting
    PromptTypes.TOT: TOT_PROMPT,
    PromptTypes.ATOMIC_FACTS: ATOMIC_PROMPT,  # is the REFERENCE LESS prompt
    PromptTypes.SCORE: SCORE_PROMPT
}
