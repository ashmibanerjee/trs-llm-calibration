import firebase_admin
from firebase_admin import credentials, firestore
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


# --- CONFIGURATION ---
# Path to your service account key file.
SERVICE_ACCOUNT_KEY_PATH = os.getcwd() + '/../../.config/rs-eval-app-3b6674ec07d6.json'
COLLECTION_PATH = 'surveys'
OUTPUT_FILE_NAME = '../../data/conv-trs/ecir-2026/human-eval/surveys_collection_raw.json'


try:
    # Initialize the Firebase Admin SDK
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    print(f"Fetching all documents from collection: {COLLECTION_PATH}...")
    print("This may take a moment and will incur read costs.")

    # Get a reference to the collection and stream the documents
    docs = db.collection(COLLECTION_PATH).stream()

    all_docs = []
    doc_count = 0
    for doc in docs:
        # Create a dictionary including the document ID and its data
        doc_data = doc.to_dict()
        doc_data['id'] = doc.id
        all_docs.append(doc_data)
        doc_count += 1

    # Write the list of documents to a local file as a nicely formatted JSON array
    with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"✅ Success! {doc_count} documents were saved to {OUTPUT_FILE_NAME}")

except Exception as e:
    print(f"❌ An error occurred: {e}")

