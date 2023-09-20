import streamlit as st
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, EmotionOptions, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns

# Creds for Natural Language Understanding
apikey = "your_key"
url = "https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/475d02ed-07a6-43f0-99fc-694ea4780a29"

# Authenticate to NLU service
authenticator = IAMAuthenticator(apikey)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',  # Use the latest version available at the time
    authenticator=authenticator
)
nlu.set_service_url(url)

# Streamlit UI
st.title("PDF Analysis with IBM Watson NLU")
st.write("Upload a PDF file for analysis.")

# File upload widget
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # CV File path (assuming it's a text file like PDF or DOCX)
    with uploaded_file:
        # Extract text from the PDF
        pdfReader = PyPDF2.PdfReader(uploaded_file)
        all_text = ""
        for pagehandle in pdfReader.pages:
            text = pagehandle.extract_text()
            all_text += text

        # Strip out unwanted text
        # Add more cleaning or processing steps as needed
        all_text = all_text.replace('o ', '')
        all_text = all_text.replace('|', '')

        # Analyze the CV text using NLU
        response = nlu.analyze(
            text=all_text,
            features=Features(entities=EntitiesOptions(), emotion=EmotionOptions(), sentiment=SentimentOptions())
        ).get_result()

        # Extract and display personal insights
        if 'emotion' in response and 'document' in response['emotion']:
            emotions = response['emotion']['document']['emotion']
            st.write("Emotions detected:")
            for emotion, score in emotions.items():
                st.write(f"{emotion.capitalize()}: {score}")

            # Visualize emotions using a bar plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=list(emotions.keys()), y=list(emotions.values()))
            plt.title("Emotions Detected in CV")
            plt.xlabel("Emotion")
            plt.ylabel("Score")
            st.pyplot(fig)  # Pass the figure to st.pyplot()

        else:
            st.write("No emotions detected in the CV.")

        # Sentiment analysis
        if 'sentiment' in response and 'document' in response['sentiment']:
            sentiment = response['sentiment']['document']
            st.write("Sentiment:")
            st.write(f"Score: {sentiment['score']}, Label: {sentiment['label']}")
        else:
            st.write("No sentiment analysis results found in the CV.")

