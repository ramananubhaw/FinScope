import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

class LLMAgent:
    """
    The LLM Agent for qualitative analysis using Google's Gemini.
    """
    def __init__(self):
        # Get API key directly from environment variables
        api_key = os.getenv("API_KEY_GEMINI")
        
        if not api_key or api_key == "YOUR_GEMINI_API_KEY":
            raise ValueError("Gemini API key not found. Please set API_KEY_GEMINI in your .env file.")
        
        genai.configure(api_key=api_key)
        
        # Use the working model that doesn't exceed quota
        self.model = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
        print("LLM Agent initialized with models/gemini-2.5-flash-preview-05-20")

    # --- THIS IS THE NEW FUNCTION YOU NEED TO ADD ---
    def generate_response(self, prompt_text: str):
        """
        Sends a generic prompt to the LLM and gets a text response.
        This is used for the final reasoning.
        """
        try:
            response = self.model.generate_content(prompt_text)
            return response.text.strip()
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return "Error generating response from LLM."
    # --- END OF NEW FUNCTION ---

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text (e.g., a news headline).
        
        Returns:
            str: 'Positive', 'Negative', or 'Neutral'.
        """
        print(f"Analyzing sentiment for: '{text[:50]}...'")
        prompt = f"""
        Analyze the sentiment of the following financial news headline.
        Classify it as ONLY 'Positive', 'Negative', or 'Neutral'.
        Return only the classification word, nothing else.

        Headline: "{text}"
        Sentiment:
        """
        try:
            # Use the generic generate_response to get the text
            response_text = self.generate_response(prompt)
            
            if response_text in ['Positive', 'Negative', 'Neutral']:
                return response_text
            else:
                # If we get something unexpected, default to Neutral
                print(f"Unexpected sentiment response: '{response_text}', defaulting to Neutral")
                return 'Neutral'
        except Exception as e:
            print(f"An error occurred during sentiment analysis: {e}")
            return "Neutral" # Default on error

    def extract_events(self, text):
        """
        Extracts key financial events from a text.
        
        Returns:
            dict: A dictionary containing extracted event info.
        """
        print(f"Extracting events from: '{text[:50]}...'")
        prompt = f"""
        From the following financial news text, identify if any of these key events are mentioned: 
        'earnings report', 'merger & acquisition', 'product launch', 'analyst upgrade', 'analyst downgrade', 'legal issues'.
        
        Respond with ONLY a JSON object with these two keys:
        1. "event_detected": boolean (true if an event was found, otherwise false)
        2. "event_type": string (the specific type of event found, or "none")

        Text: "{text}"
        """
        try:
            # Use the generic generate_response to get the text
            response_text = self.generate_response(prompt)
            
            # Try to parse JSON response
            try:
                event_data = json.loads(response_text)
                return event_data
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract manually
                print(f"JSON parsing failed, response was: {response_text}")
                return {"event_detected": False, "event_type": "none"}
                
        except Exception as e:
            print(f"An error occurred during event extraction: {e}")
            return {"event_detected": False, "event_type": "none"}

    def safe_analyze_sentiment(self, text, max_retries=3):
        """
        A safer version with retry logic for sentiment analysis.
        """
        for attempt in range(max_retries):
            try:
                return self.analyze_sentiment(text)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    continue
                else:
                    print(f"All {max_retries} attempts failed: {e}")
                    return "Neutral"