# DataGenie 🧞‍♂️

Welcome to DataGenie, your ultimate companion for expediting the Exploratory Data Analysis (EDA) process. Built on the robust Streamlit framework and enhanced with the intelligent capabilities of Langchain's OpenAI backend, DataGenie offers a user-friendly platform designed to make your data analytics journey magical.

## Features
- **Efficient EDA:** Rapidly understand your data with powerful, automated analysis.
- **User-Friendly:** Easy-to-navigate interface, suitable for beginners and experts alike.
- **Powered by AI:** Leverage the latest AI technologies for deep insights and predictions.
- **Open Source:** Free to use, modify, and distribute.

## Getting Started
Follow these simple steps to set up DataGenie and start transforming your data into actionable insights:

### Step 1: Clone the Repository
```bash
git clone https://github.com/jeffersonqiu/DataGenie.git
```

### Step 2: Install Dependencies
Navigate to the cloned repository's directory and install the required dependencies using Poetry:
```bash
cd DataGenie
poetry install
```

### Step 3: Configure Environment Variables
Rename the .env.example file to .env and update it with your OpenAI API key (find your API key at OpenAI Platform):
```python
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 4: Run DataGenie
Launch DataGenie by running the Streamlit application:
```python
poetry run streamlit run app.py
```

## What to use for
1. **Explore Your Data**: Upload your datasets and let DataGenie guide you through the EDA process.<br>
2. **Generate Insights**: Utilize the power of AI to uncover patterns, trends, and correlations.

## Contributing
We welcome contributions to DataGenie! Whether it's adding new features, improving documentation, or reporting issues, your help makes DataGenie better for everyone.

## Acknowledgments
- Streamlit for the incredible app framework.
- Langchain and OpenAI for the AI backend that powers our analyses.

Dive into your data with DataGenie and uncover the insights you've been searching for. Happy analyzing!


