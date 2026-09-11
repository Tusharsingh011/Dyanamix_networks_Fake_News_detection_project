
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------
# STEP 0: NLTK stopwords download (ek baar chalega)
# ---------------------------------------------------
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


# ---------------------------------------------------
# STEP 1: Excel dataset load karna
# ---------------------------------------------------
def load_data(file_path):
    
    file_path = "data\\raw\\fake.csv"
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------
# STEP 2: 'type' column ko Real/Fake mein map karna
# ---------------------------------------------------
def map_label(type_value):
    """
    Dataset ke 'type' column ki categories ko binary label mein convert karta hai.

    Fake  -> bs, conspiracy, hate, junksci, satire, fake
    Real  -> state, bias (comparatively factual/formal sources)
    """
    fake_categories = ['bs', 'conspiracy', 'hate', 'junksci', 'satire', 'fake']

    if pd.isna(type_value):
        return None  # missing type wali rows baad mein drop karenge

    type_value = str(type_value).strip().lower()

    if type_value in fake_categories:
        return 'fake'
    else:
        return 'real'


# ---------------------------------------------------
# STEP 3: Text cleaning function
# ---------------------------------------------------
def clean_text(text):
    """
    Ek single text string ko clean karta hai:
    - lowercase
    - URLs remove
    - punctuation remove
    - numbers remove
    - extra spaces remove
    - stopwords remove
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # URLs hatao
    text = re.sub(r'http\S+|www\S+', '', text)

    # Punctuation hatao
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Numbers hatao
    text = re.sub(r'\d+', '', text)

    # Extra spaces hatao
    text = re.sub(r'\s+', ' ', text).strip()

    # Stopwords hatao
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]

    return ' '.join(words)


# ---------------------------------------------------
# STEP 4: Poora preprocessing pipeline (ek saath sab chalana)
# ---------------------------------------------------
def preprocess_pipeline(input_path, output_path):
    """
    Poora process:
    1. Data load
    2. Sirf zaroori columns rakho (title, text, type)
    3. Label map karo (real/fake)
    4. Title + text ko combine + clean karo
    5. Processed data ko CSV mein save karo
    """
    # 1. Load
    df = load_data(input_path)

    # 2. Sirf zaroori columns select karo
    df = df[['title', 'text', 'type']].copy()

    # 3. Label mapping
    df['label'] = df['type'].apply(map_label)

    # Missing/invalid label wali rows drop karo
    df = df.dropna(subset=['label'])

    # 4. Title aur text ko combine karke ek 'content' column banao
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + ' ' + df['text']

    # Cleaning apply karo
    print("Cleaning text... isme thoda time lag sakta hai bade dataset ke liye.")
    df['clean_text'] = df['content'].apply(clean_text)

    # Khaali clean_text wali rows hatao
    df = df[df['clean_text'].str.strip() != '']

    # Final columns rakho
    final_df = df[['clean_text', 'label']]

    # 5. Save karo
    final_df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    print(f"Final shape: {final_df.shape}")
    print(f"Label distribution:\n{final_df['label'].value_counts()}")
 
    return final_df


# ---------------------------------------------------
# Directly run karne ke liye (test ke liye)
# ---------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE = "data/raw/news_dataset.csv"       # apni actual file ka naam daalo
    OUTPUT_FILE = "data/processed/cleaned_data.csv"

    preprocess_pipeline(INPUT_FILE, OUTPUT_FILE)
