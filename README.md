# **Sign Language Translator**

## **📌 Overview**

The Sign Language Translator is an AI-powered application that helps translate American Sign Language (ASL) alphabet gestures in real-time. It uses a pre-trained model from Hugging Face to recognize ASL letters from images, camera input, and live video feeds.

## **🚀 Features**

📸 Image Load: Upload an image to detect ASL alphabet gestures.

📷 Take Picture: Capture a picture using your camera and classify the gesture.

🎥 Live ASL Mode: Enable your webcam for real-time ASL alphabet recognition.

📖 About Us: Information about the project and its purpose.

📞 Contact Us: Dummy phone number, LinkedIn, Facebook, Instagram links, and email.

💬 Feedback: A section to collect user feedback.

## **🏗 Technologies Used**

Python 🐍

Streamlit 🎨 (for the interactive UI)

Hugging Face Transformers 🤗 (for ASL classification)

OpenCV 📷 (for handling live video input)

Torch 🔥 (for deep learning model inference)

## **🛠 Installation & Setup**

***Clone the repository:***
git clone https://github.com/your-username/sign-language-translator.git
cd sign-language-translator

***Create a virtual environment*** (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

***Install dependencies:****
pip install -r requirements.txt

***Run the application:****
streamlit run app.py

## **📄 Project Structure**

📂 sign-language-translator
│── 📜 app.py               # Main application script
│── 📜 requirements.txt      # List of dependencies
│── 📜 README.md             # Project documentation (this file)

## **🔥 How It Works**

Load an image, take a picture, or enable live mode.

The AI model processes the input and classifies the ASL gesture.

The recognized letter is displayed to the user.
