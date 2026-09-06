#cd
cd NLP-Blooms-Taxonomy-main
cd ai-multimodal-question-generator

#checkin the folder content
dir

# Create the virtual environment
python -m venv venv

# If PowerShell restricts running scripts, temporarily allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the environment
.\venv\Scripts\Activate.ps1

#pip installments
pip install --upgrade pip
pip install -r requirements.txt

#if needed for nltk
python -c "import nltk; nltk.download('popular')"

#running the App
streamlit run app.py
