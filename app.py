import streamlit as st
import pandas as pd
import google.generativeai as genai
from tempfile import NamedTemporaryFile
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import csv
import speech_recognition as sr
import re
from io import StringIO


# ==================== Data Analysis Functions ====================

def generate_response(model, query: str, df: pd.DataFrame) -> str:
    """Generate a response using Gemini Pro."""
    # Create a context string with dataset information
    context = f"""
    Dataset Information:
    - Columns: {', '.join(df.columns)}
    - Number of rows: {len(df)}
    - Data sample:
    {df.head().to_string()}

    Numeric summary:
    {df.describe().to_string()}
    """

    prompt = f"""
    Based on the following dataset:
    {context}

    Please answer this question: {query}

    If the question involves calculations, show the steps and provide numerical insights.
    If relevant, mention if a visualization would be helpful for better understanding.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


def create_visualization(df: pd.DataFrame, query: str) -> Optional[plt.Figure]:
    """Create relevant visualization based on the query and data."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine the type of visualization based on the query
        query_lower = query.lower()

        # Distribution analysis
        if any(word in query_lower for word in ['distribution', 'spread', 'histogram']):
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Find the most relevant numeric column based on the query
                target_col = numeric_cols[0]
                for col in numeric_cols:
                    if col.lower() in query_lower:
                        target_col = col
                        break
                sns.histplot(data=df, x=target_col, ax=ax)
                plt.title(f'Distribution of {target_col}')
                return fig

        # Correlation analysis
        elif any(word in query_lower for word in ['correlation', 'relationship', 'compare']):
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                plt.title('Correlation Heatmap')
                return fig

        # Time series or trend analysis
        elif any(word in query_lower for word in ['trend', 'time', 'over time']):
            if 'date' in df.columns or 'year' in df.columns:
                time_col = next(col for col in df.columns if 'date' in col.lower() or 'year' in col.lower())
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    plt.plot(df[time_col], df[numeric_cols[0]])
                    plt.title(f'{numeric_cols[0]} over {time_col}')
                    plt.xticks(rotation=45)
                    return fig

        # Group comparison
        elif any(word in query_lower for word in ['group', 'category', 'department', 'compare']):
            categorical_cols = df.select_dtypes(include=['object']).columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                sns.barplot(data=df, x=cat_col, y=num_col, ax=ax)
                plt.title(f'{num_col} by {cat_col}')
                plt.xticks(rotation=45)
                return fig

        return None
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


def process_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Process the uploaded CSV file with enhanced error handling.
    Returns a tuple of (DataFrame, error_message)
    """
    try:
        # Read the file content
        file_content = uploaded_file.read()

        # Detect the file encoding
        encoding_result = chardet.detect(file_content)
        encoding = encoding_result['encoding']

        # Try to detect the delimiter
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(file_content.decode(encoding))
        delimiter = dialect.delimiter

        # Create a temporary file
        with NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        # Try reading with detected parameters
        try:
            df = pd.read_csv(tmp_file_path, encoding=encoding, delimiter=delimiter)
        except:
            # Fallback to common delimiters if detection fails
            for delim in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(tmp_file_path, encoding=encoding, delimiter=delim)
                    break
                except:
                    continue
            else:
                return None, "Could not determine the correct delimiter for the CSV file"

        # Clean up temp file
        os.unlink(tmp_file_path)

        # Verify the DataFrame has content
        if df.empty:
            return None, "The CSV file appears to be empty"

        if len(df.columns) == 0:
            return None, "No columns found in the CSV file"

        # Check if we have only one column (might indicate parsing issues)
        if len(df.columns) == 1 and df.columns[0].count(',') > 0:
            return None, "File appears to be improperly parsed. Please check the CSV format"

        return df, None

    except UnicodeDecodeError:
        return None, "Error decoding the file. Please ensure it's a valid CSV file with proper encoding"
    except csv.Error:
        return None, "Error reading the CSV file. Please check if it's properly formatted"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ==================== Speech Recognition Functions ====================

def recognize_speech() -> str:
    """Capture speech and return recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening...")
        audio = recognizer.listen(source)

    try:
        st.info("Recognizing...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio")
        return ""
    except sr.RequestError:
        st.error("Could not request results from the speech recognition service")
        return ""


# ==================== Data Wrangling Functions ====================

def detect_file_encoding_and_delimiter(file_content: bytes) -> Tuple[str, str]:
    """Detect file encoding and delimiter with enhanced error handling."""
    # Detect encoding
    encoding_result = chardet.detect(file_content)
    encoding = encoding_result['encoding']

    # Try to detect delimiter
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(file_content.decode(encoding)[:1024])  # Only sample first 1KB
        delimiter = dialect.delimiter
    except:
        # Fallback to common delimiters
        for delim in [',', ';', '\t', '|']:
            try:
                pd.read_csv(StringIO(file_content.decode(encoding)), delimiter=delim, nrows=1)
                delimiter = delim
                break
            except:
                continue
        else:
            delimiter = ','  # Ultimate fallback

    return encoding, delimiter


def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load data from uploaded file with robust error handling."""
    try:
        file_content = uploaded_file.read()
        encoding, delimiter = detect_file_encoding_and_delimiter(file_content)

        # Read into DataFrame
        df = pd.read_csv(StringIO(file_content.decode(encoding)), delimiter=delimiter)

        # Clean column names (remove extra whitespace)
        df.columns = [str(col).strip() for col in df.columns]

        # Basic validation
        if df.empty:
            return None, "The file appears to be empty"
        if len(df.columns) == 1 and df.columns[0].count(',') > 0:
            return None, "Possible delimiter issue - only one column detected"

        return df, None

    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def find_best_column_match(df, search_term: str) -> Optional[str]:
    """Find the best matching column name with flexible matching."""
    if not isinstance(search_term, str):
        return None

    search_term = search_term.lower().strip(" '\"")  # Clean the search term

    # First try exact match (case insensitive)
    for col in df.columns:
        if search_term == str(col).lower().strip():
            return col

    # Then try partial match
    for col in df.columns:
        if search_term in str(col).lower().strip():
            return col

    return None


def extract_entities(command: str) -> dict:
    """Extract entities from natural language command with improved parsing."""
    command = command.lower().strip()
    entities = {
        'columns': [],
        'actions': [],
        'values': [],
        'conditions': []
    }

    # Handle special commands
    if command in ['show columns', 'list columns']:
        return {'actions': ['show_columns']}

    # Handle rename commands (multiple patterns)
    rename_patterns = [
        r"rename\s+'?([\w\s]+)'?\s+to\s+'?([\w\s]+)'?",  # rename id to emp_no
        r"rename\s+column\s+'?([\w\s]+)'?\s+to\s+'?([\w\s]+)'?",  # rename column id to emp_no
        r"change\s+'?([\w\s]+)'?\s+to\s+'?([\w\s]+)'?",  # change id to emp_no
        r"rename\s+the\s+'?([\w\s]+)'?\s+column\s+to\s+'?([\w\s]+)'?"  # rename the id column to emp_no
    ]

    for pattern in rename_patterns:
        match = re.search(pattern, command)
        if match:
            entities['columns'] = [match.group(1).strip(), match.group(2).strip()]
            entities['actions'] = ['rename']
            return entities

    # Handle drop commands
    drop_patterns = [
        r"drop\s+'?([\w\s]+)'?\s+column",  # drop id column
        r"remove\s+'?([\w\s]+)'?\s+column",  # remove id column
        r"delete\s+'?([\w\s]+)'?\s+column"  # delete id column
    ]

    for pattern in drop_patterns:
        match = re.search(pattern, command)
        if match:
            entities['columns'] = [match.group(1).strip()]
            entities['actions'] = ['drop']
            return entities

    # Handle missing values
    if 'missing' in command or 'null' in command:
        if 'drop' in command:
            entities['actions'] = ['drop_missing']
            return entities
        elif 'fill' in command:
            # Try to extract fill value
            value_match = re.search(r"fill\s+(?:missing|null)\s+(?:with|as)\s+([\w]+)", command)
            if value_match:
                entities['values'] = [value_match.group(1)]
            entities['actions'] = ['fill_missing']
            return entities

    # Handle filter commands
    filter_patterns = [
        r"keep\s+rows\s+where\s+'?([\w\s]+)'?\s+([><=]+)\s+([\d]+)",  # keep rows where age > 30
        r"filter\s+for\s+'?([\w\s]+)'?\s+([><=]+)\s+([\d]+)"  # filter for salary > 50000
    ]

    for pattern in filter_patterns:
        match = re.search(pattern, command)
        if match:
            entities['columns'] = [match.group(1).strip()]
            entities['actions'] = ['filter']
            entities['conditions'] = [f"{match.group(2)}{match.group(3)}"]
            return entities

    # Default case (show help)
    entities['actions'] = ['help']
    return entities


def execute_data_command(df: pd.DataFrame, command: str) -> Tuple[pd.DataFrame, str]:
    """Execute natural language data wrangling command with improved feedback."""
    original_shape = df.shape
    entities = extract_entities(command)
    response = []

    try:
        # Handle show columns command
        if 'show_columns' in entities['actions']:
            return df, f"Columns in dataset:\n{', '.join(df.columns)}"

        # Handle rename commands
        if 'rename' in entities['actions']:
            if len(entities['columns']) >= 2:
                old_name = entities['columns'][0]
                new_name = entities['columns'][1]

                actual_old_name = find_best_column_match(df, old_name)
                if actual_old_name:
                    df.rename(columns={actual_old_name: new_name}, inplace=True)
                    response.append(f"Renamed column '{actual_old_name}' to '{new_name}'")
                else:
                    # Try to suggest similar column names
                    similar_cols = [col for col in df.columns if old_name.lower() in col.lower()]
                    if similar_cols:
                        response.append(f"Column '{old_name}' not found. Did you mean: {', '.join(similar_cols)}?")
                    else:
                        response.append(f"Column '{old_name}' not found. Available columns: {', '.join(df.columns)}")
            else:
                response.append("Need both old and new names for rename operation (e.g., 'rename id to emp_no')")

        # Handle drop commands
        elif 'drop' in entities['actions']:
            if entities['columns']:
                for col in entities['columns']:
                    actual_col = find_best_column_match(df, col)
                    if actual_col:
                        df.drop(columns=[actual_col], inplace=True)
                        response.append(f"Dropped column '{actual_col}'")
                    else:
                        similar_cols = [col for col in df.columns if col.lower() in col.lower()]
                        if similar_cols:
                            response.append(f"Column '{col}' not found. Did you mean: {', '.join(similar_cols)}?")
                        else:
                            response.append(f"Column '{col}' not found. Available columns: {', '.join(df.columns)}")
            else:
                response.append("Please specify which column to drop (e.g., 'drop id column')")

        # Handle drop missing values
        elif 'drop_missing' in entities['actions']:
            initial_count = len(df)
            df.dropna(inplace=True)
            dropped = initial_count - len(df)
            response.append(f"Dropped {dropped} rows with missing values")

        # Handle fill missing values
        elif 'fill_missing' in entities['actions']:
            if entities['values']:
                fill_value = entities['values'][0]
                try:
                    fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
                except ValueError:
                    pass  # Keep as string if not a number

                df.fillna(fill_value, inplace=True)
                response.append(f"Filled missing values with '{fill_value}'")
            else:
                response.append("Please specify fill value (e.g., 'fill missing with 0')")

        # Handle filter commands
        elif 'filter' in entities['actions']:
            if entities['columns'] and entities['conditions']:
                col = find_best_column_match(df, entities['columns'][0])
                if col:
                    try:
                        condition = entities['conditions'][0]
                        if '>' in condition:
                            value = float(condition.replace('>', ''))
                            df = df[df[col] > value]
                        elif '<' in condition:
                            value = float(condition.replace('<', ''))
                            df = df[df[col] < value]
                        elif '=' in condition:
                            value = float(condition.replace('=', ''))
                            df = df[df[col] == value]
                        response.append(f"Filtered to keep rows where {col} {condition}")
                    except:
                        response.append(f"Could not filter on column '{col}' with condition '{condition}'")
                else:
                    response.append(f"Column '{entities['columns'][0]}' not found")
            else:
                response.append("Please specify filter condition (e.g., 'keep rows where age > 30')")

        # Help command
        elif 'help' in entities['actions']:
            help_text = """
            Available commands:
            - Rename columns: 'rename id to emp_no', 'change column age to customer_age'
            - Drop columns: 'drop id column', 'remove salary column'
            - Handle missing data: 'drop rows with missing values', 'fill missing with 0'
            - Filter data: 'keep rows where age > 30', 'filter for salary > 50000'
            - Show columns: 'show columns', 'list columns'
            """
            return df, help_text

        # Generate summary of changes
        if response:
            final_shape = df.shape
            if original_shape != final_shape:
                response.append(f"\nData shape changed from {original_shape} to {final_shape}")
        else:
            response.append("No valid operations detected in command")

        return df, '\n'.join(response)

    except Exception as e:
        return df, f"Error executing command: {str(e)}"


# ==================== Main Application ====================

def main():
    st.set_page_config(page_title="Data Analysis Suite", layout="wide")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose the app mode",
                                ["Data Analysis Chatbot", "Data Wrangling Assistant"])

    if app_mode == "Data Analysis Chatbot":
        run_data_analysis_app()
    else:
        run_data_wrangling_app()


def run_data_analysis_app():
    st.title("CSV Data Analysis Chatbot with Speech Recognition")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

    # Setup Gemini Pro
    try:
        genai.configure(api_key=os.getenv('AIzaSyCrPR2EvuFuzt2SYMU6PMjD0kLjsohxSXU'))
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
    except Exception as e:
        st.error(f"Error initializing Gemini Pro: {str(e)}")
        st.info("Please make sure you have set the GOOGLE_API_KEY environment variable.")
        return

    # File upload section
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        if not st.session_state.file_uploaded:
            df, error_message = process_csv(uploaded_file)

            if error_message:
                st.error(error_message)
                st.info("""
                Common fixes:
                1. Make sure your CSV file is not empty
                2. Check if the file is properly formatted with headers
                3. Verify the file encoding (UTF-8 is recommended)
                4. Ensure the delimiter is consistent throughout the file
                5. Try opening and resaving the file in a text editor
                """)
            else:
                st.session_state.df = df
                st.session_state.file_uploaded = True
                st.success("File uploaded successfully!")
                st.write("Preview of your data:")
                st.dataframe(df.head())
                st.info(f"Detected {len(df.columns)} columns: {', '.join(df.columns)}")

        # Only show chat interface if file is successfully uploaded
        if st.session_state.file_uploaded:
            st.subheader("Ask questions about your data")

            # Create a clean separation between file upload and chat
            st.markdown("---")

            # Add the text input box with a clear placeholder
            user_query = st.text_input(
                "Type your question here:",
                placeholder="Example: What is the average salary by department?",
                key="user_input"
            )

            if user_query:  # This checks if the user entered text
                with st.spinner("Analyzing..."):
                    response = generate_response(model, user_query, st.session_state.df)

                    # Add to chat history
                    st.session_state.chat_history.append(("You", user_query))
                    st.session_state.chat_history.append(("Assistant", response))

                    # Create visualization if appropriate
                    fig = create_visualization(st.session_state.df, user_query)
                    if fig:
                        st.pyplot(fig)

            # Add the speech recognition button
            if st.button("Start Speaking"):
                query = recognize_speech()
                if query:
                    st.text_area("Recognized Speech", value=query, height=150)

                    # Generate the response from speech input
                    if query:
                        with st.spinner("Analyzing..."):
                            response = generate_response(model, query, st.session_state.df)

                            # Add to chat history
                            st.session_state.chat_history.append(("You", query))
                            st.session_state.chat_history.append(("Assistant", response))

                            # Create visualization if appropriate
                            fig = create_visualization(st.session_state.df, query)
                            if fig:
                                st.pyplot(fig)

            # Display chat history
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for role, message in st.session_state.chat_history:
                    if role == "You":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**Assistant:** {message}")
                        st.markdown("---")

            # Add a button to clear chat history
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()

            # Add a button to upload a new file
            if st.button("Upload New File"):
                st.session_state.file_uploaded = False
                st.session_state.df = None
                st.session_state.chat_history = []
                st.experimental_rerun()


def run_data_wrangling_app():
    st.title("NLP Data Wrangling Assistant")
    st.write("Upload your dataset and modify it using natural language commands!")

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'history' not in st.session_state:
        st.session_state.history = []

    # File upload section
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])

    if uploaded_file is not None and st.session_state.df is None:
        df, error = load_data(uploaded_file)
        if error:
            st.error(f"Error loading file: {error}")
        else:
            st.session_state.df = df
            st.success("Dataset loaded successfully!")
            st.write("Data preview:")
            st.dataframe(df.head())
            st.write(f"Columns: {', '.join(df.columns)}")

    if st.session_state.df is not None:
        st.subheader("Data Wrangling Commands")

        # Display current data info
        st.write(f"Current data shape: {st.session_state.df.shape}")

        # Command input
        command = st.text_input(
            "Enter your command (e.g., 'rename id to emp_no', 'drop missing values')",
            placeholder="Try 'show columns' to start"
        )

        if st.button("Execute Command"):
            with st.spinner("Processing command..."):
                st.session_state.df, result = execute_data_command(st.session_state.df, command)
                st.session_state.history.append((command, result))
                st.success("Command executed!")
                st.write(result)
                st.write("Updated data preview:")
                st.dataframe(st.session_state.df.head())

        # Command history
        if st.session_state.history:
            st.subheader("Command History")
            for cmd, res in st.session_state.history:
                st.markdown(f"**Command:** `{cmd}`")
                st.markdown(f"**Result:** {res}")
                st.markdown("---")

        # Data download
        st.subheader("Export Data")
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download modified data as CSV",
            data=csv,
            file_name='modified_data.csv',
            mime='text/csv'
        )

        # Reset button
        if st.button("Reset Data"):
            st.session_state.df = None
            st.session_state.history = []
            st.experimental_rerun()


if __name__ == "__main__":
    main()