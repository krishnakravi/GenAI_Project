import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import pandas as pd
import json
import customtkinter as ctk
from PIL import Image, ImageTk
import speech_recognition as sr
import pytesseract
import cv2
import webbrowser
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the existing project modules
from phase1 import run_phase1
from phase2 import implement_chain_of_thought
from phase3 import build_rag_system
from phase1setup import check_and_install_dependencies

class ExerciseRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise Chatbot")
        self.root.geometry("400x600")  # Phone-like aspect ratio
        
        # Set theme
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")
        
        # Initialize BLIP model and processor for image recognition
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Create widgets first
        self.create_widgets()
        
        # Initialize RAG system after widgets are created
        try:
            self.rag_system = build_rag_system()
            self.add_message("System: RAG system initialized successfully.", "system")
        except Exception as e:
            self.add_message(f"System: Failed to initialize RAG system: {e}", "system")
            self.rag_system = None
        
        # Conversation history
        self.conversation_history = []
        
        # Variable to store the image for preview
        self.current_image = None
        
    def create_widgets(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Chat window
        self.chat_frame = ctk.CTkFrame(self.main_frame)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat display using tk.Text
        self.chat_display = tk.Text(self.chat_frame, height=20, wrap=tk.WORD, bg="#2b2b2b", fg="white")
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.tag_configure("user", foreground="lightgreen", font=("Arial", 10))
        self.chat_display.tag_configure("bot", foreground="cyan", font=("Arial", 10, "bold"))
        self.chat_display.tag_configure("system", foreground="gray", font=("Arial", 9, "italic"))
        
        # Input section
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Input type selection
        self.input_type_var = ctk.StringVar(value="Text")
        self.input_type_combo = ctk.CTkComboBox(
            self.input_frame,
            values=["Text", "Voice", "Image"],
            variable=self.input_type_var,
            command=self.update_input_method
        )
        self.input_type_combo.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Text input
        self.text_input = ctk.CTkTextbox(self.input_frame, height=50)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Send button
        self.send_btn = ctk.CTkButton(
            self.input_frame,
            text="Send",
            command=self.send_message
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Voice and Image buttons
        self.record_voice_btn = ctk.CTkButton(
            self.input_frame,
            text="Record Voice",
            command=self.record_voice
        )
        self.upload_image_btn = ctk.CTkButton(
            self.input_frame,
            text="Upload Image",
            command=self.upload_image
        )
        self.record_voice_btn.pack_forget()
        self.upload_image_btn.pack_forget()
        
        # Image preview section
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.pack(fill=tk.X, padx=10, pady=5)
        self.image_label = tk.Label(self.image_frame, text="No image uploaded yet.")
        self.image_label.pack()

    def update_input_method(self, choice):
        """Update input method and reset button layout cleanly."""
        self.record_voice_btn.pack_forget()
        self.upload_image_btn.pack_forget()
        
        if choice == "Voice":
            self.record_voice_btn.pack(side=tk.LEFT, padx=5, pady=5)
        elif choice == "Image":
            self.upload_image_btn.pack(side=tk.LEFT, padx=5, pady=5)
        # Text mode doesn't need additional buttons

    def add_message(self, message, msg_type):
        """Add a message to the chat display."""
        self.chat_display.insert(tk.END, f"{message}\n", msg_type)
        self.chat_display.see(tk.END)

    def send_message(self):
        """Send the current input as a message and get a response."""
        text = self.text_input.get("1.0", tk.END).strip()
        if text and self.rag_system:  # Check if RAG is initialized
            # Add user message to chat
            self.add_message(f"You: {text}", "user")
            self.conversation_history.append({"role": "user", "content": text})
            
            try:
                # Get response from RAG system
                response = self.rag_system.answer_query(text)
                if isinstance(response, dict):
                    answer = response.get('answer', 'No information available.')
                    self.add_message(f"Bot: {answer}", "bot")
                    self.conversation_history.append({"role": "bot", "content": answer})
                else:
                    self.add_message("Bot: Error: Invalid response format", "bot")
            except Exception as e:
                self.add_message(f"Bot: Error: {e}", "bot")
            
            # Clear input after sending
            self.text_input.delete("1.0", tk.END)
        elif not self.rag_system:
            self.add_message("System: RAG system is not initialized. Please check dependencies.", "system")

    def record_voice(self):
        """Record audio and transcribe it to text."""
        def record_and_transcribe():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                self.add_message("System: Listening...", "system")
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    self.text_input.delete("1.0", tk.END)
                    self.text_input.insert(tk.END, text)
                    self.add_message("System: Voice transcribed successfully.", "system")
                except sr.UnknownValueError:
                    self.add_message("System: Could not understand audio.", "system")
                except sr.RequestError:
                    self.add_message("System: Speech recognition service error.", "system")
        
        threading.Thread(target=record_and_transcribe, daemon=True).start()

    def upload_image(self):
        """Upload an image, recognize the exercise, and display steps using RAG."""
        def process_image():
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
            if file_path:
                try:
                    image = Image.open(file_path).convert("RGB")
                    
                    # Generate caption using BLIP model
                    inputs = self.processor(images=image, return_tensors="pt")
                    outputs = self.model.generate(**inputs)
                    caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Update chat display with caption
                    self.add_message(f"System: Image recognized: {caption}", "system")
                    
                    # Construct a query to get exercise steps
                    query = f"How to perform {caption.lower().replace('a man', 'I')}"
                    self.add_message(f"System: Generating exercise steps...", "system")
                    
                    # Get exercise steps from RAG system
                    if self.rag_system:
                        response = self.rag_system.answer_query(query)
                        if isinstance(response, dict):
                            answer = response.get('answer', 'No exercise steps found in the dataset.')
                            self.add_message(f"Bot: Here's how to perform this exercise:\n{answer}", "bot")
                        else:
                            self.add_message("Bot: Could not generate exercise steps. Please try a different query.", "bot")
                    else:
                        self.add_message("System: RAG system is not initialized. Please check dependencies.", "system")
                    
                    # Clear text input
                    self.text_input.delete("1.0", tk.END)
                    
                    # Resize image for preview
                    image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    self.current_image = ImageTk.PhotoImage(image)
                    self.image_label.configure(image=self.current_image, text="")
                except Exception as e:
                    self.add_message(f"System: Error processing image: {str(e)}", "system")
        
        threading.Thread(target=process_image, daemon=True).start()

def main():
    check_and_install_dependencies()
    root = ctk.CTk()
    app = ExerciseRecommendationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()