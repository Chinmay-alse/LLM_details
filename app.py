import streamlit as st
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Model Analysis Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .css-1d391kg {
        padding: 1rem 1rem 1.5rem;
    }
    .stSelectbox label {
        font-size: 1.1rem;
        font-weight: 500;
        color: #0f2537;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin: 0.5rem 0;
    }
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .header-style {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0f2537;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader-style {
        font-size: 1.5rem;
        font-weight: 500;
        color: #0f2537;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_csv(file):
    """Load and return dataframe from CSV file"""
    df = pd.read_csv(file)
    return df

def get_csv_info(df):
    """Get basic information about the CSV file"""
    info = {
        'columns': list(df.columns),
        'rows': len(df),
        'summary': df.describe().to_string(),
        'full_data': df.to_string()  # Include full dataset
    }
    return info

def get_model_capabilities(df_row):
    """Get detailed capabilities for a model based on its metrics"""
    capabilities = []
    
    # MMLU (Massive Multitask Language Understanding)
    mmlu = float(df_row['MMLU'].strip('%')) if isinstance(df_row.get('MMLU'), str) else 0
    if mmlu > 85:
        capabilities.append("Strong general knowledge and reasoning")
    
    # GPQA (General Purpose Question Answering)
    gpqa = float(df_row['GPQA'].strip('%')) if isinstance(df_row.get('GPQA'), str) and '~' not in df_row['GPQA'] else 0
    if gpqa > 70:
        capabilities.append("Excellent question answering ability")
    
    # HumanEval (Code Generation)
    humaneval = float(df_row['HumanEval'].strip('%')) if isinstance(df_row.get('HumanEval'), str) and df_row['HumanEval'] != '-' else 0
    if humaneval > 85:
        capabilities.append("Strong code generation capabilities")
    
    # Math
    math_score = float(df_row['Math'].strip('%')) if isinstance(df_row.get('Math'), str) else 0
    if math_score > 80:
        capabilities.append("Advanced mathematical reasoning")
    
    # MMMU (Massive Multitask Multimodal Understanding)
    mmmu = float(df_row['MMMU'].strip('%')) if isinstance(df_row.get('MMMU'), str) else 0
    if mmmu > 80:
        capabilities.append("Strong multimodal understanding")
        
    # MGSM (Machine Generated Story Matching)
    mgsm = float(df_row['MGSM'].strip('%')) if isinstance(df_row.get('MGSM'), str) else 0
    if mgsm > 80:
        capabilities.append("Excellent story understanding and generation")
        
    # DocVQA (Document Visual Question Answering)
    docvqa = float(df_row['DocVQA'].strip('%')) if isinstance(df_row.get('DocVQA'), str) else 0
    if docvqa > 80:
        capabilities.append("Strong document visual understanding")
        
    # Mathvista
    mathvista = float(df_row['Mathvista'].strip('%')) if isinstance(df_row.get('Mathvista'), str) else 0
    if mathvista > 80:
        capabilities.append("Advanced visual mathematical reasoning")
    
    # Multimodal Capabilities
    if isinstance(df_row.get('Multimodal Capabilities'), str):
        capabilities.append(f"Supports: {df_row['Multimodal Capabilities']}")
    
    return capabilities

def create_benchmark_graph(df, selected_model):
    """Create a single graph showing all metrics for the selected model"""
    metrics = ['MMLU', 'GPQA', 'HumanEval', 'Math', 'MMMU', 'MGSM', 'DocVQA', 'Mathvista']
    df_plot = df.copy()
    
    # Clean and convert percentage strings to float values
    for metric in metrics:
        if metric in df_plot.columns:
            df_plot[metric] = df_plot[metric].apply(lambda x: float(str(x).strip('%').replace('~', '')) if pd.notnull(x) and str(x).strip() != '-' else 0)
        else:
            df_plot[metric] = 0  # Initialize missing columns with zeros
    
    # Filter data for selected model
    model_data = df_plot[df_plot['Model Name'] == selected_model]
    
    # Prepare data for plotting
    y_values = []
    hover_texts = []
    for metric in metrics:
        value = model_data[metric].iloc[0] if metric in model_data.columns and len(model_data) > 0 else 0
        y_values.append(value)
        hover_texts.append(f"{metric}: {value:.1f}%")
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add bars for each metric with custom colors
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=y_values,
        marker_color=colors,
        text=[f"{val:.1f}%" for val in y_values],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    # Update layout with clean styling and smooth transitions
    fig.update_layout(
        title={
            'text': f"<b>Performance Metrics for {selected_model}</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        showlegend=False,
        height=500,
        width=800,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=12),
        margin=dict(t=100, b=50, l=50, r=50),
        uirevision=True,  # This maintains the zoom/pan state
        transition_duration=500  # Smooth transition duration in milliseconds
    )
    
    # Update axes
    fig.update_yaxes(
        range=[0, 100],
        title_text="Score (%)",
        gridcolor='lightgray',
        griddash='dash',
        zeroline=True,
        zerolinecolor='darkgray',
        zerolinewidth=1,
        ticksuffix='%',
        showline=True,
        linewidth=1,
        linecolor='darkgray',
        mirror=True
    )
    
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='darkgray',
        mirror=True,
        tickangle=45  # Angled labels for better readability
    )

    return fig

def generate_response(question, df_info, df):
    """Generate response using Groq API"""
    
    # Create a metrics dictionary for direct lookups
    metrics_info = {
        'MMLU': "Massive Multitask Language Understanding - Tests general knowledge and reasoning across multiple domains",
        'GPQA': "General Purpose Question Answering - Measures model's ability to answer diverse questions accurately",
        'HumanEval': "Evaluates code generation capabilities and programming skills",
        'Math': "Assesses mathematical reasoning abilities and problem-solving skills",
        'MMMU': "Massive Multitask Multimodal Understanding - Evaluates model's ability to understand and process multimodal tasks",
        'MGSM': "Machine Generated Story Matching - Evaluates model's story understanding and generation capabilities",
        'DocVQA': "Document Visual Question Answering - Tests ability to answer questions about documents with visual elements",
        'Mathvista': "Assesses mathematical reasoning abilities with visual inputs and diagrams"
    }
    
    # Check if the question is asking about a specific metric
    for metric, description in metrics_info.items():
        if metric.lower() in question.lower():
            return f"{metric}: {description}"
    
    # If not asking about a specific metric, provide the full analysis
    # Get model capabilities for better context
    model_capabilities = []
    for _, row in df.iterrows():
        capabilities = get_model_capabilities(row)
        if capabilities:
            model_capabilities.append(f"{row['Model Name']}: {', '.join(capabilities)}")
    
    context = f"""
    I have a CSV file containing AI model benchmarks with the following information:
    
    Columns: {df_info['columns']}
    Number of rows: {df_info['rows']}
    
    Key Metrics Included:
    - MMLU (Massive Multitask Language Understanding): Tests general knowledge and reasoning
    - GPQA (General Purpose Question Answering): Measures question answering ability
    - HumanEval: Evaluates code generation capabilities
    - Math: Assesses mathematical reasoning abilities
    - MMMU (Massive Multitask Multimodal Understanding): Evaluates model's ability to understand and process multimodal tasks
    - MGSM (Machine Generated Story Matching): Evaluates model's story understanding and generation
    - DocVQA (Document Visual Question Answering): Tests ability to answer questions about documents with visual elements
    - Mathvista: Assesses mathematical reasoning with visual inputs
    
    Model Capabilities Summary:
    {chr(10).join(model_capabilities)}
    
    Full Data:
    {df_info['full_data']}
    
    Statistical Summary:
    {df_info['summary']}
    
    Question: {question}
    
    Please provide a detailed analysis based on the data provided, considering the various benchmarks and their implications for model performance.
    When comparing models, consider:
    1. Overall performance across all metrics
    2. Specialized strengths (e.g., math, coding, general knowledge, multimodal understanding)
    3. Multimodal capabilities and MMMU performance
    4. Best-suited use cases
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an AI model analyst expert. For questions about specific metrics, provide concise, focused answers.
                For comparison questions or general analysis, provide detailed responses considering multiple metrics."""
            },
            {
                "role": "user",
                "content": context
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.1
    )
    
    return chat_completion.choices[0].message.content

def main():
    # Custom header with icon
    st.markdown('<p class="header-style">🤖 AI Model Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Introduction text
    st.markdown("""
    <div style='background-color: #e8f4f9; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        <p style='margin:0; font-size: 1.1rem;'>
            Welcome to the AI Model Analysis Dashboard! Upload your CSV file containing model benchmarks 
            and explore performance metrics across different models. Ask questions about the data and 
            get AI-powered insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File upload with custom styling
    st.markdown('<p class="subheader-style">📁 Upload Data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load and process the data
        df = load_csv(uploaded_file)
        df_info = get_csv_info(df)

        # Data preview in an expandable section
        with st.expander("📋 View Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True)

        # Model selector and visualization
        st.markdown('<p class="subheader-style">📈 Model Performance Analysis</p>', unsafe_allow_html=True)
        
        # Create two columns for model selection
        select_col1, select_col2 = st.columns([2, 1])
        
        with select_col1:
            selected_model = st.selectbox(
                "Select a model to analyze:",
                options=df['Model Name'].tolist(),
                help="Choose a model to view its detailed performance metrics"
            )

        # Display benchmark graph
        fig = create_benchmark_graph(df, selected_model)
        st.plotly_chart(fig, use_container_width=True)

        # Question and Answer section
        st.markdown('<p class="subheader-style">❓ Ask Questions</p>', unsafe_allow_html=True)
        
        question = st.text_input(
            "Ask a question about the models:",
            placeholder="e.g., 'Which model has the best MMLU score?' or 'Compare the math capabilities of different models'",
            help="Ask any question about the models and their performance metrics"
        )

        if question:
            if not os.getenv("GROQ_API_KEY"):
                st.error("⚠️ Please set your GROQ_API_KEY in the .env file")
            else:
                with st.spinner("🤔 Analyzing your question..."):
                    response = generate_response(question, df_info, df)
                    st.markdown(f"""
                        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;'>
                            <p style='margin:0; white-space: pre-wrap;'>{response}</p>
                        </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
