import streamlit as st
from langchain.prompts import PromptTemplate, PipelinePromptTemplate
from langchain_community.callbacks import StreamlitCallbackHandler

first_look_prompt = '''
    {salutation}, You need to explore the dataframe in a few indicated steps below. Please indicate clearly what is the steps being done.
    1. Data Overview: 
    1.1. Show first five rows of the data
    1.2. Show the columns name
    1.3. Show the missing values and duplicated for each column
    1.4. Show Data summary: df.describe()
    1.5. Calculate correlation in the data
    1.6. Identify potential outliers
    1.7. Identify potential new features to include
'''

first_look_template = PromptTemplate.from_template(first_look_prompt)

def text_runner(_agent, df, text):
    st.write(text)
    st.write(_agent.run(text))

def function_runner(_agent, text, function):
    st.write(text)
    st.write(function)

@st.cache_data
def first_look_function(df, _agent):
    st.write('**Data Overview**')
    text_runner(_agent, df, "Show columns name")
    text_runner(_agent, df, "Show the missing values and duplicated for each column")
    function_runner(_agent, "Show data summary", df.describe())
    text_runner(_agent, df, "Identify potential outliers")
    text_runner(_agent, df, "Identify potential new features to include")

    return None


sb_template = PromptTemplate.from_template(
    "Output simple one liner steps for: {question}"
)

eda_template = '''
{intro}

{do_not_list}

{dataframe_description}
'''
eda_prompt = PromptTemplate.from_template(eda_template)

intro_eda_template = '''
Give me step by step idea for an EDA provided that this is the details of the dataframe. 
The answer should be in bullet form, each step should be less than 5 words. 
Example format of the list (start with '-', ends with '.'): 
- Identify missing values.
'''

do_not_eda_template = '''
- Do not show backend work such as import libraries, load dataframe.
- Do not provide the answer to the EDA, i.e. x columns, y rows. 
- Do not provide any suggestion related to visualization.
- Provide not more than 8 concrete/ not repetitive steps.
- Do not show Feature Engineering steps
- Do not generate something that we couldn't answer based on the existing dataframe, i.e. corr values when there is no numerical columns in the dataframe
'''

dataframe_description_template = '''
Here is the details of the dataframe: {dataframe_details}
'''

intro_eda_prompt = PromptTemplate.from_template(intro_eda_template)
do_not_eda_prompt = PromptTemplate.from_template(do_not_eda_template)
dataframe_description_eda_prompt = PromptTemplate.from_template(dataframe_description_template)

input_eda_prompts = [
    ("intro", intro_eda_prompt),
    ("do_not_list", do_not_eda_prompt),
    ("dataframe_description", dataframe_description_eda_prompt),
]

filled_eda_prompt = PipelinePromptTemplate(
    final_prompt=eda_prompt, pipeline_prompts=input_eda_prompts
)

@st.cache_data
def eda_selection_generator(_eda_chain, _df_details):
    return _eda_chain.invoke({'dataframe_details': _df_details})['text']

@st.cache_data
def individual_eda(_pd_agent, _eda_selected, peda_click_count):
    st_callback = StreamlitCallbackHandler(st.container())
    st.write(_pd_agent.run(_eda_selected, callbacks=[st_callback]))


aaa_template = '''
{intro}

{dataframe_description}

{do_not_list}
'''
aaa_prompt = PromptTemplate.from_template(aaa_template)

# Give me a list of possible questions that Pandas agent can answer well about the dataframe.
intro_aaa_template = '''

Each sentence should be less than 6 words long and clear. 
Provide not more than 8 concrete/ not repetitive questions.
'''

dataframe_description_aaa_template = '''
Here is the details of the dataframe: {dataframe_details}
'''

do_not_aaa_template = '''
- DO NOT provide any list that is already captured before in the double quotation "{eda_selection}".
- Do not provide list that cannot be answered by pandas agent.
- Do not provide questions about number of rows/ columns, missing values
'''

intro_aaa_prompt = PromptTemplate.from_template(intro_aaa_template)
dataframe_description_aaa_prompt = PromptTemplate.from_template(dataframe_description_aaa_template)
do_not_aaa_prompt = PromptTemplate.from_template(do_not_aaa_template)

input_aaa_prompts = [
    ("intro", intro_aaa_prompt),
    ("dataframe_description", dataframe_description_aaa_prompt),
    ("do_not_list", do_not_aaa_prompt),
]

filled_aaa_prompt = PipelinePromptTemplate(
    final_prompt=aaa_prompt, pipeline_prompts=input_aaa_prompts
)

@st.cache_data
def aaa_sample_generator(_aaa_chain, _dataframe_details, _eda_selection):
    return _aaa_chain.invoke({'dataframe_details': _dataframe_details, 'eda_selection': _eda_selection})['text']

@st.cache_data
def aaa_answer_generator(_pd_agent, _user_prompt, refreshed):
    st_callback = StreamlitCallbackHandler(st.container())
    answer_to_user = _pd_agent.run(_user_prompt, callbacks=[st_callback])
    st.write(answer_to_user)