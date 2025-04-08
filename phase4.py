import os
import speech_recognition as sr
import pytesseract
from PIL import Image
from transformers import pipeline

# Initialize recognizer and OCR
recognizer = sr.Recognizer()
# Update the Tesseract path based on your system (e.g., Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Load pre-trained model for text processing (e.g., question-answering)
text_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

def process_text_input(query):
    """
    Process text input and return a response using a pre-trained model.
    
    Args:
        query (str): User's text query
        
    Returns:
        str: Response to the query
    """
    # Example context (replace with your job data or knowledge base)
    context = "Data scientists need skills in statistics, programming (Python, R), and data visualization (Tableau). " \
              "Experience with machine learning and SQL is also valuable."
    try:
        result = text_model(question=query, context=context)
        return result['answer']
    except Exception as e:
        return f"Error processing text: {str(e)}"

def process_voice_input():
    """
    Convert voice input to text and process it.
    
    Returns:
        str: Response based on voice input
    """
    with sr.Microphone() as source:
        print("Listening for voice input...")
        audio = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Voice Input Detected: {text}")
            return process_text_input(text)
        except sr.UnknownValueError:
            return "Sorry, I couldn’t understand the audio."
        except sr.RequestError:
            return "Sorry, there was an error with the speech recognition service."
        except Exception as e:
            return f"Error processing voice: {str(e)}"

def process_image_input(image_path):
    """
    Extract text from an image and process it.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text or processed result
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        print(f"Extracted Text from Image: {text}")
        # For demonstration, return the extracted text; you can further process it with process_text_input
        return text.strip() if text.strip() else "No text detected in the image."
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Text input example
    text_query = "What skills are needed for data scientists?"
    print("Text Response:", process_text_input(text_query))
    
    # Voice input example
    print("Voice Response:", process_voice_input())
    
    # Image input example
    image_path = "path/to/resume.png"  # Replace with an actual image path
    print("Image Response:", process_image_input(image_path))