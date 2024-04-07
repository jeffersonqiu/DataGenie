import streamlit as st
import os
import pandas as pd
# from utils import clicked, describe_dataframe, to_show, checkbox_clicked, additional_clicked_fun
# from llm import first_look_function, eda_selection_generator, individual_eda, aaa_sample_generator, aaa_answer_generator
# from llm import filled_eda_prompt, filled_aaa_prompt
from langchain.prompts import PromptTemplate, PipelinePromptTemplate
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# # utils
# def clicked(button):
#     st.session_state.clicked[button] = True

# def checkbox_clicked(button):
#     st.session_state.checkbox_menu[button] = st.session_state.checkbox_menu[button] == False

# def additional_clicked_fun(button):
#     st.session_state.refreshed[button] += 1

# @st.cache_data
# def describe_dataframe(df):
#     # Initialize a list to hold descriptions for each column
#     column_descriptions = []
    
#     for column in df.columns:
#         # Basic column data
#         col_type = df[column].dtype
#         num_nulls = df[column].isnull().sum()
#         null_info = "has some missing values" if num_nulls > 0 else "has no missing values"
        
#         # Detailed stats for numeric columns
#         if pd.api.types.is_numeric_dtype(df[column]):
#             max_value = df[column].max()
#             min_value = df[column].min()
#             mean_value = df[column].mean()
#             column_descriptions.append(f"{column} (numeric) - type: {col_type}, {null_info}, max: {max_value}, min: {min_value}, mean: {mean_value:.2f}")
#         # Add more conditions for other data types (e.g., categorical, datetime) as needed
#         else:
#             column_descriptions.append(f"{column} - type: {col_type}, {null_info}")
    
#     # Combine all column descriptions into a single string
#     detailed_description = "; ".join(column_descriptions)
    
#     overall_description = f"The dataset has {len(df)} rows and {len(df.columns)} columns. Column details: {detailed_description}."
    
#     return overall_description

# @st.cache_data
# def to_show(df, show_selected, rows_to_show):
#     switch_dic = {
#         'First few rows': df.head(rows_to_show), 'Last few rows': df.tail(rows_to_show), 'Random':df.sample(rows_to_show)
#     }
#     st.write(f'There are {len(df)} rows and {len(df.columns)} columns.')
#     # columns = [col for col in df.columns]
#     # st.write('Column Names')
#     # st.write(columns)

#     return switch_dic[show_selected]

# # llm
# first_look_prompt = '''
#     {salutation}, You need to explore the dataframe in a few indicated steps below. Please indicate clearly what is the steps being done.
#     1. Data Overview: 
#     1.1. Show first five rows of the data
#     1.2. Show the columns name
#     1.3. Show the missing values and duplicated for each column
#     1.4. Show Data summary: df.describe()
#     1.5. Calculate correlation in the data
#     1.6. Identify potential outliers
#     1.7. Identify potential new features to include
# '''

# first_look_template = PromptTemplate.from_template(first_look_prompt)

# def text_runner(_agent, df, text):
#     st.write(text)
#     st.write(_agent.run(text))

# def function_runner(_agent, text, function):
#     st.write(text)
#     st.write(function)

# @st.cache_data
# def first_look_function(df, _agent):
#     st.write('**Data Overview**')
#     text_runner(_agent, df, "Show columns name")
#     text_runner(_agent, df, "Show the missing values and duplicated for each column")
#     function_runner(_agent, "Show data summary", df.describe())
#     text_runner(_agent, df, "Identify potential outliers")
#     text_runner(_agent, df, "Identify potential new features to include")

#     return None


# sb_template = PromptTemplate.from_template(
#     "Output simple one liner steps for: {question}"
# )

# eda_template = '''
# {intro}

# {do_not_list}

# {dataframe_description}
# '''
# eda_prompt = PromptTemplate.from_template(eda_template)

# intro_eda_template = '''
# Give me step by step idea for an EDA provided that this is the details of the dataframe. 
# The answer should be in bullet form, each step should be less than 5 words. 
# Example format of the list (start with '-', ends with '.'): 
# - Identify missing values.
# '''

# do_not_eda_template = '''
# - Do not show backend work such as import libraries, load dataframe.
# - Do not provide the answer to the EDA, i.e. x columns, y rows. 
# - Do not provide any suggestion related to visualization.
# - Provide not more than 8 concrete/ not repetitive steps.
# - Do not show Feature Engineering steps
# '''

# dataframe_description_template = '''
# Here is the details of the dataframe: {dataframe_details}
# '''

# intro_eda_prompt = PromptTemplate.from_template(intro_eda_template)
# do_not_eda_prompt = PromptTemplate.from_template(do_not_eda_template)
# dataframe_description_eda_prompt = PromptTemplate.from_template(dataframe_description_template)

# input_eda_prompts = [
#     ("intro", intro_eda_prompt),
#     ("do_not_list", do_not_eda_prompt),
#     ("dataframe_description", dataframe_description_eda_prompt),
# ]

# filled_eda_prompt = PipelinePromptTemplate(
#     final_prompt=eda_prompt, pipeline_prompts=input_eda_prompts
# )

# @st.cache_data
# def eda_selection_generator(_eda_chain, _df_details):
#     return _eda_chain.invoke({'dataframe_details': _df_details})['text']

# @st.cache_data
# def individual_eda(_pd_agent, _eda_selected, peda_click_count):
#     st_callback = StreamlitCallbackHandler(st.container())
#     st.write(_pd_agent.run(_eda_selected, callbacks=[st_callback]))


# aaa_template = '''
# {intro}

# {dataframe_description}

# {do_not_list}
# '''
# aaa_prompt = PromptTemplate.from_template(aaa_template)

# # Give me a list of possible questions that Pandas agent can answer well about the dataframe.
# intro_aaa_template = '''

# Each sentence should be less than 6 words long and clear. 
# Provide not more than 8 concrete/ not repetitive questions.
# '''

# dataframe_description_aaa_template = '''
# Here is the details of the dataframe: {dataframe_details}
# '''

# do_not_aaa_template = '''
# - DO NOT provide any list that is already captured before in the double quotation "{eda_selection}".
# - Do not provide list that cannot be answered by pandas agent.
# - Do not provide questions about number of rows/ columns, missing values
# '''

# intro_aaa_prompt = PromptTemplate.from_template(intro_aaa_template)
# dataframe_description_aaa_prompt = PromptTemplate.from_template(dataframe_description_aaa_template)
# do_not_eda_prompt = PromptTemplate.from_template(do_not_aaa_template)

# input_aaa_prompts = [
#     ("intro", intro_aaa_prompt),
#     ("dataframe_description", dataframe_description_aaa_prompt),
#     ("do_not_list", do_not_eda_prompt),
# ]

# filled_aaa_prompt = PipelinePromptTemplate(
#     final_prompt=aaa_prompt, pipeline_prompts=input_aaa_prompts
# )

# @st.cache_data
# def aaa_sample_generator(_aaa_chain, _dataframe_details, _eda_selection):
#     return _aaa_chain.invoke({'dataframe_details': _dataframe_details, 'eda_selection': _eda_selection})['text']

# @st.cache_data
# def aaa_answer_generator(_pd_agent, _user_prompt):
#     st_callback = StreamlitCallbackHandler(st.container())
#     answer_to_user = _pd_agent.invoke(_user_prompt, callbacks=[st_callback])
#     st.write(answer_to_user['output'])

# # from dotenv import load_dotenv
# # load_dotenv()

# # Initialization
# if 'clicked' not in st.session_state:
#     st.session_state.clicked = {'begin_button': False}

# if 'eda_selection' not in st.session_state:
#     st.session_state.eda_selection = []

# if 'data_exist' not in st.session_state:
#     st.session_state.data_exist = False

# if 'checkbox_menu' not in st.session_state:
#     st.session_state.checkbox_menu = {
#         'show_data_button': True,
#         'eda_button': True,
#         'va_button': True,
#         'aaa_button': True
#     }

# if 'column_names' not in st.session_state:
#     st.session_state.column_names = None

# if 'df_details' not in st.session_state:
#     st.session_state.df_details = None

# if 'peda_clicked' not in st.session_state:
#     st.session_state.peda_clicked = 0

# st.set_page_config(page_title="DataGenie", page_icon="🧞‍♂️")

# st.markdown("<h1 style='text-align: center;'>Data Genie 🧞‍♂️</h1>", unsafe_allow_html=True)
# st.text('I am your helpful Data Analyst AI. Feel free to drop your data for me to analyse.')

# # LLM Declaration
# try:
#     USER_OPENAI_API_KEY = st.text_input('*Please input your OpenAI API Key: (sk-xxxx)*', placeholder="Your API key here")
#     llm = ChatOpenAI(model_name='gpt-4-0125-preview', api_key=USER_OPENAI_API_KEY)
#     col1, col2, col3 = st.columns(3)
#     with col2:
#         st.button("Let's begin the magic", on_click=clicked, args=['begin_button'])
# except:
#     st.markdown("""
#     <style>
#     .red-italic-text {
#         color: red;
#         font-style: italic;
#         text-align: center;
#     }
#     </style>
#     <div class='red-italic-text'>Please first add a valid OpenAI API Code</div>
#     """, unsafe_allow_html=True)


# if st.session_state.clicked['begin_button']:
#     with st.expander('Upload your .csv data here'):
#         data = st.file_uploader(' ', type ='csv')
#     if data is not None:
#         st.session_state.data_exist = True
#         df = pd.read_csv(data, low_memory=False)
#         st.session_state.column_names = df.columns


#         pd_agent = create_pandas_dataframe_agent(llm, df, verbose=True)
#         if st.session_state.checkbox_menu['show_data_button']:
#             st.divider()
#             st.subheader('Show Data')
            
#             show_selection = ['First few rows', 'Last few rows', 'Random']
#             show_selected = st.selectbox('Select type of EDA to perform on this dataset:', options=show_selection)
#             rows_to_show = st.number_input('How many rows to show?', format='%d', step=1, value = 5)
#             st.write(to_show(df, show_selected, rows_to_show))

#         if st.session_state.checkbox_menu['eda_button']:
#             st.divider()
#             st.subheader('Exploratory Data Analysis')

#             df_details = describe_dataframe(df)
#             st.session_state.df_details = df_details

#             eda_chain = LLMChain(llm=llm, prompt=filled_eda_prompt)

#             eda_selection = eda_selection_generator(eda_chain, df_details)
#             st.session_state.eda_selection = eda_selection

#             eda_list = eda_selection.split('.\n-')[1:]
#             eda_list.insert(0, '[Default] Perform default EDA')
            
#             st.markdown('#### EDA to Perform')

#             eda_selected = st.selectbox('Based on the dataframe, here are the most common EDA steps to perform:', options=eda_list)
            
#             if st.button('Perform EDA', on_click=additional_clicked_fun):
#                 prompt = PromptTemplate.from_template(eda_selected)
#                 with st.chat_message('assistant'):
#                     if eda_selected != '[Default] Perform default EDA':
#                         individual_eda(pd_agent, eda_selected, st.session_state.peda_clicked)
#                     else:
#                         first_look_function(df, pd_agent)
                    
#         # if st.session_state.checkbox_menu['va_button']:
#         #     st.divider()
#         #     st.subheader("Variable of Interest")
#         #     user_question = st.selectbox("What variable are you interested in exploring?", options = st.session_state.column_names)
#         #     st.write(user_question)

            

#         if st.session_state.checkbox_menu['aaa_button']:
#             st.divider()
#             st.subheader("Ask AI Anything")
#             st.write('Hint: Check sidebar for Prompt Inspiration')
#             user_prompt = st.text_area('Enter your question here!')
#             if user_prompt:
#                 aaa_answer_generator(pd_agent, user_prompt)


# with st.sidebar:
#     if st.session_state.clicked['begin_button']:
#         st.header('Guide')
#         st.write('1. To begin, enter data in .csv format.')
#         if st.session_state.data_exist == True:
#             st.write('2. Choose what do you want to do?')
#             show_data_button = st.checkbox('Show Data', True, on_change=checkbox_clicked, args=['show_data_button'])
#             eda_button = st.checkbox('Exploratory Data Analysis', True, on_change=checkbox_clicked, args=['eda_button'])
                
#             # va_button = st.checkbox('Variable Analysis', True, on_change=checkbox_clicked, args=['va_button'])
#             aaa_button = st.checkbox('Ask AI Anything!', True, on_change=checkbox_clicked, args=['aaa_button'])

#             st.divider()
#             if show_data_button:
#                 with st.expander('Columns Names'):
#                     st.markdown("Navigation: [Show Data](#show-data)", unsafe_allow_html=True)
#                     st.subheader('Columns Names')
#                     st.write(st.session_state.column_names)

#             if eda_button:
#                 if len(st.session_state.eda_selection) != 0:
#                     with st.expander('EDA: Suggested Steps'):
#                         st.markdown("Navigation: [EDA](#exploratory-data-analysis)", unsafe_allow_html=True)
#                         st.button("Refresh EDA Suggestions", on_click=additional_clicked_fun)
#                         st.write(st.session_state.eda_selection)


#             if aaa_button:
#                 with st.expander('Prompt Inspiration'):
#                     st.markdown("Navigation: [Ask AI Anything](#ask-ai-anything)", unsafe_allow_html=True)
#                     aaa_chain = LLMChain(llm=llm, prompt=filled_aaa_prompt)
#                     _dataframe_details = st.session_state.df_details
#                     _eda_selection = st.session_state.eda_selection
#                     aaa_samples = aaa_sample_generator(aaa_chain, _dataframe_details, _eda_selection)
#                     # st.write(llm.invoke('Give me a list of possible questions that Pandas agent can answer well about the dataframe'))
#                     st.write(aaa_samples)
            
    

        
