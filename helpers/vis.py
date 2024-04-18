from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import (
    PromptTemplate
)
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser
import streamlit as st
import importlib

def prompt_generator_chart_type():  
    system_template = """
        The following is a conversation between a Human and an AI assistant expert on data visualization with perfect Python 3 syntax. The human will provide a sample dataset for the AI to use as the source. The real dataset that the human will use with the response of the AI is going to have several more rows. The AI assistant will only reply in the following JSON format: 

        {{ 
        "charts": [{{'title': string, 'chartType': string, 'parameters': {{...}}}}, ... ]
        }}

        Instructions:

        1. chartType must only contain methods of plotly.express from the Python library Plotly.
        2. The format for charType string: plotly.express.chartType.
        3. For each chartType, parameters must contain the value to be used for all parameters of that plotly.express method.
        4. There should 4 parameters for each chart.
        5. Do not include "data_frame" in the parameters.
        6. Features in 'parameters' should not contain a space character. Joining more than one word should be done by using '_'.
        7. There should be {num_charts} charts in total.
        """
    system_message_prompt = PromptTemplate.from_template(system_template)

    human_template = """
        Human: 
        This is the dataset:

        {data}

        Create chart that analyze this specific topic: {topic}
        """
    human_message_prompt = PromptTemplate.from_template(human_template)

    full_template = """{system_prompt}

    {human_prompt}
    """
    full_prompt = PromptTemplate.from_template(full_template)

    input_prompts = [
        ("system_prompt", system_message_prompt),
        ("human_prompt", human_message_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts, input_variables=['num_charts','data', 'topic']
    )
    return pipeline_prompt


def prompt_generator_feature_engineering():
    system_template = """
        Instructions:
        1. Read the visualization specs as given to you. Check on all variables in 'parameters'.
        2. If any of the variables in 'parameters' does not appear as a column in the original dataset, return pandas function which transforms the original dataset into a new dataset containing ALL variables in parameters.
        3. Return this pandas operations in string form. Only return the string to execute without any explanation! 
        4. If there are >1 line of code, split them with ';'
        5. Sometimes you need to rename the column to ensure ALL variables in 'parameters' are represented exactly in the final_df dataset. 
        6. Always end the answer with 'final_df = df'

        Assumptions:
        1. Assume that original dataframe is given as 'df'
        2. Assume that the columns in the original dataframe might not have the right dtypes. Adjust it first to accept the right dtypes.

        Do not do this:
        1. Do not use python``` code here ``` format. Directly return pandas function in text format.
        """
    system_message_prompt = PromptTemplate.from_template(system_template)

    human_template = """
        Human: 
        This is the dataset:
        {data}
        Please perform sorting of the data!

        This is the column names in the original dataset:
        {column_names}

        This is the visualization specs: 
        {vis_specs}
        """
    human_message_prompt = PromptTemplate.from_template(human_template)

    full_template = """{system_prompt}

    {human_prompt}
    """
    full_prompt = PromptTemplate.from_template(full_template)

    input_prompts = [
        ("system_prompt", system_message_prompt),
        ("human_prompt", human_message_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts, input_variables=['data', 'column_names', 'vis_specs']
    )
    return pipeline_prompt

def chart_generator(llm, df, user_question_vis):
    chart_type_chain = LLMChain(llm=llm, 
                    prompt=prompt_generator_chart_type(), 
                    output_parser=JsonOutputParser(), 
                    output_key='vis_specs'
                    )

    chart_types = chart_type_chain.run({
        "data":df.head(10),
        "topic": user_question_vis,
        "num_charts": 1
    })

    return chart_types['charts']

def vis_generator(chart, llm, df):
    params = chart['parameters']
    fe_chain = LLMChain(llm=llm, prompt=prompt_generator_feature_engineering(), output_key='final_output')
    fe_code = fe_chain.run({
        "data": df.head(10),
        "column_names": df.columns,
        "vis_specs": chart
    })
    # st.write(fe_code)
    final_df = None
    try:
        exec(fe_code)
        st.write('Successfully Executed Feature Engineering Script')
        final_df = df
    except Exception as e:
        st.write(f"Error during Feature Engineering Execution: {e}")

    if final_df is not None:
        # st.write(df.head())
        # st.write(final_df.head())  # Using .head() to display just the first few rows
        params['data_frame'] = final_df

        chart_type = chart['chartType']
        px_module = importlib.import_module("plotly.express")
        chart_function = getattr(px_module, chart_type.split('.')[-1])  
        fig = chart_function(**params)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("final_df was not defined.")