import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel
import speech_recognition as sr
from PIL import Image
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder for RAG system (replace with actual import if available)
try:
    from phase3 import build_rag_system
except ImportError:
    def build_rag_system():
        return None
    logger.warning("Phase 3 RAG system not found. RAG functionality will be disabled.")

class ExerciseRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise Recommendation App")
        self.root.geometry("600x500")

        # Initialize fine-tuned BERT model and tokenizer
        self.bert_model = None
        self.bert_tokenizer = None
        self.label2id = {}
        self.id2label = {}
        self.load_bert_model_and_labels()

        # Initialize BLIP VQA model
        self.blip_processor = None
        self.blip_model = None
        self.load_blip_model()

        # Initialize RAG system (from Phase 3)
        self.rag_system = build_rag_system()
        if self.rag_system:
            logger.info("RAG system initialized successfully.")
        else:
            logger.warning("RAG system not available.")

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

        # Create GUI widgets
        self.create_widgets()

    def load_bert_model_and_labels(self):
        """Load the fine-tuned BERT model, tokenizer, and label mappings."""
        try:
            # Load labels from dataset
            dataset_path = "data/processed_exercises.csv"
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                unique_types = df['Type'].dropna().unique().tolist()
                self.label2id = {label: idx for idx, label in enumerate(unique_types)}
                self.id2label = {idx: label for label, idx in self.label2id.items()}
                logger.info(f"Loaded labels: {self.label2id}")
            else:
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            # Load tokenizer and model
            self.bert_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
            self.bert_model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
            self.bert_model.eval()
            logger.info("Fine-tuned BERT model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load BERT model or labels: {str(e)}")
            self.bert_model = None
            self.bert_tokenizer = None

    def load_blip_model(self):
        """Load the BLIP VQA model and processor."""
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            self.blip_model.eval()
            logger.info("BLIP VQA model loaded successfully.")
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to load BLIP model: {str(e)}. Image processing will be disabled.")
            self.blip_processor = None
            self.blip_model = None

    def classify_exercise(self, text):
        """Classify an exercise description using the fine-tuned BERT model."""
        if not self.bert_model or not self.bert_tokenizer:
            return "BERT model not loaded."

        try:
            # Tokenize input
            inputs = self.bert_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )

            # Move to CPU/GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                predicted_id = torch.argmax(logits, dim=1).item()

            return self.id2label.get(predicted_id, "Unknown")
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return f"Error classifying exercise: {str(e)}"

    def process_voice_input(self):
        """Convert voice to text using speech recognition."""
        try:
            with sr.Microphone() as source:
                self.add_message("System: Listening for voice input...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                self.add_message(f"System: Recognized: {text}")
                return text
        except sr.UnknownValueError:
            self.add_message("System: Could not understand audio.")
            return None
        except sr.RequestError as e:
            self.add_message(f"System: Speech recognition error: {str(e)}")
            return None
        except Exception as e:
            self.add_message(f"System: Voice processing error: {str(e)}")
            return None

    def process_image_input(self, image_path):
        """Detect exercise in an image using BLIP VQA."""
        if not self.blip_processor or not self.blip_model:
            self.add_message("System: Image processing disabled due to missing BLIP model.")
            return None
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            question = "What exercise is the person doing?"
            inputs = self.blip_processor(images=image, text=question, return_tensors="pt")

            # Move to CPU/GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.blip_model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate answer
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs)
                answer = self.blip_processor.decode(outputs[0], skip_special_tokens=True)

            if answer.strip():
                self.add_message(f"System: Detected exercise: {answer}")
                return answer.strip()
            else:
                self.add_message("System: No exercise detected in image.")
                return None
        except Exception as e:
            self.add_message(f"System: Image processing error: {str(e)}")
            return None

    def create_widgets(self):
        """Create GUI components."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            height=20,
            wrap=tk.WORD,
            state='disabled'
        )
        self.chat_display.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input frame
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Text input
        self.text_input = ttk.Entry(self.input_frame)
        self.text_input.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.input_frame.columnconfigure(0, weight=1)

        # Buttons
        self.text_button = ttk.Button(
            self.input_frame,
            text="Classify Text",
            command=self.handle_text_input
        )
        self.text_button.grid(row=0, column=1, padx=5)

        self.voice_button = ttk.Button(
            self.input_frame,
            text="Voice Input",
            command=self.handle_voice_input
        )
        self.voice_button.grid(row=0, column=2, padx=5)

        self.image_button = ttk.Button(
            self.main_frame,
            text="Upload Image",
            command=self.handle_image_input
        )
        self.image_button.grid(row=2, column=0, columnspan=3, pady=5)

        # Configure resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

    def add_message(self, message):
        """Add a message to the chat display."""
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')

    def process_input(self, text):
        """Process input text with classification and optional RAG."""
        if not text:
            return

        self.add_message(f"You: {text}")

        # Classify with fine-tuned BERT model
        prediction = self.classify_exercise(text)
        self.add_message(f"App: Predicted exercise type: {prediction}")

        # Get RAG response if available
        if self.rag_system:
            try:
                rag_response = self.rag_system.answer_query(text)
                self.add_message(f"App (RAG): {rag_response.get('answer', 'No response')}")
            except Exception as e:
                self.add_message(f"App: RAG error: {str(e)}")

    def handle_text_input(self):
        """Handle text input from entry field."""
        user_input = self.text_input.get().strip()
        if user_input:
            self.text_input.delete(0, tk.END)
            self.process_input(user_input)

    def handle_voice_input(self):
        """Handle voice input."""
        text = self.process_voice_input()
        if text:
            self.process_input(text)

    def handle_image_input(self):
        """Handle image input."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            text = self.process_image_input(file_path)
            if text:
                self.process_input(text)

def main():
    root = tk.Tk()
    app = ExerciseRecommendationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()