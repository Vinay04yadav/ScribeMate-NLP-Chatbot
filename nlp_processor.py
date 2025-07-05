import spacy
import string
import nltk
from collections import Counter
from rake_nltk import Rake
from transformers import pipeline
import torch
import textwrap # Good to keep for potential future use
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch # Already likely added for summarizer
import random
import re


# Global variable for spaCy model (load once)
nlp_spacy = None # Initialize as None
print("INFO: Attempting to load spaCy model 'en_core_web_sm'...")
try:
    # Attempt to load the model installed in the local environment
    nlp_spacy = spacy.load('en_core_web_sm')
    print("INFO: SpaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    # Model not found - user needs to download it via terminal
    print("ERROR: SpaCy model 'en_core_web_sm' not found.")
    print("       Please ensure you have run 'python -m spacy download en_core_web_sm'")
    print("       in your activated 'venv' terminal environment.")
    nlp_spacy = None # Ensure it remains None if loading failed
except Exception as e:
    # Catch other potential errors during loading
    print(f"ERROR: An unexpected error occurred loading spaCy model: {e}")
    nlp_spacy = None


# Initialize Rake globally
rake_nltk_var = None # Initialize as None
print("INFO: Attempting to initialize RAKE...")
try:
    # Rake uses NLTK stopwords, assumes relevant NLTK data is available
    rake_nltk_var = Rake()
    print("INFO: RAKE initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize RAKE: {e}")
    print("       Ensure NLTK stopwords are downloaded if needed ('nltk.download(\"stopwords\")').")
    rake_nltk_var = None # Ensure it remains None if failed

# Global variables for summarizer pipeline
summarizer_pipeline = None
summarizer_model_name = "t5-small" # Default model

def initialize_summarizer():
    """Initializes the summarization pipeline globally if not already loaded."""
    global summarizer_pipeline # Indicate modification of global variable
    if summarizer_pipeline is None:
        print(f"INFO: Initializing summarization pipeline with '{summarizer_model_name}'...")
        try:
            # Determine device (GPU or CPU)
            device_to_use = 0 if torch.cuda.is_available() else -1
            summarizer_pipeline = pipeline("summarization", model=summarizer_model_name, device=device_to_use)
            print(f"INFO: Summarization pipeline initialized successfully on {'GPU' if device_to_use == 0 else 'CPU'}.")
        except ImportError:
             print(f"ERROR: Required library for {summarizer_model_name} (e.g., sentencepiece) might be missing. Please install it.")
             summarizer_pipeline = None
        except Exception as e:
            # Catch other potential errors (model download issues, memory errors)
            print(f"ERROR: Failed to initialize summarization pipeline: {e}")
            summarizer_pipeline = None # Ensure it's None if failed
    # else:
        # print("INFO: Summarization pipeline already loaded.") # Optional: uncomment for debugging
    return summarizer_pipeline


# Global variables for QG components
qg_model_name = "potsawee/t5-large-generation-squad-QuestionAnswer" # Default model
qg_tokenizer = None
qg_model = None
qg_device = None

def initialize_qg_components():
    """Initializes the QG tokenizer and model globally if not already loaded."""
    global qg_tokenizer, qg_model, qg_device, qg_model_name # Indicate modification of global variables
    if qg_model is None or qg_tokenizer is None:
        print(f"INFO: Initializing QG components with '{qg_model_name}'...")
        try:
            # Determine device (GPU or CPU)
            qg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
            qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name).to(qg_device)
            print(f"INFO: QG components initialized successfully on {qg_device}.")
            return True # Indicate success
        except Exception as e:
            print(f"ERROR: Failed to initialize QG components '{qg_model_name}': {e}")
            # Reset globals on failure to prevent partial state
            qg_tokenizer = None
            qg_model = None
            qg_device = None
            return False # Indicate failure
    else:
         # If already loaded, just ensure model is on the right device
         # (This might be important if device availability changes between runs, though unlikely in stable env)
         try:
             current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             if qg_device != current_device:
                  qg_device = current_device
                  qg_model.to(qg_device)
                  print(f"INFO: Moved existing QG model to {qg_device}.")
             # else: print("INFO: QG components already loaded and on correct device.") # Optional debug
         except Exception as e:
              print(f"ERROR: Failed checking/moving existing QG model device: {e}")
              # Potentially reset them if device check fails badly? Safer to leave loaded.
         return True # Still return True if they were loaded, even if device check had issues









def preprocess_text(raw_text: str) -> tuple | None:
    """
    Cleans, tokenizes, and lemmatizes the input text using spaCy.

    Args:
        raw_text: The raw lecture transcript string.

    Returns:
        A tuple containing:
        - cleaned_text (str): Lowercased, stripped text.
        - doc (spacy.Doc): The processed spaCy Doc object.
        - sentences (list): A list of spaCy sentence spans.
        - lemmatized_tokens (list): A list of filtered, lemmatized tokens (strings).
        Returns None if input is invalid or spaCy model fails.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        print("Error: Input text must be a non-empty string.")
        return None

    # Use print statements for logging/debugging, keep them for now
    print("INFO: Preprocessing - Cleaning text...")
    cleaned_text = raw_text.lower().strip()

    print("INFO: Preprocessing - Processing with spaCy...")
    try:
        # Use the globally loaded spaCy model
        global nlp_spacy # Make sure Python knows we're using the global var
        if nlp_spacy is None:
             print("ERROR: SpaCy model is not loaded. Cannot preprocess.")
             return None
        doc = nlp_spacy(cleaned_text)
    except Exception as e:
        print(f"ERROR: Failed processing text with spaCy: {e}")
        return None

    print("INFO: Preprocessing - Extracting sentences...")
    sentences = list(doc.sents)

    print("INFO: Preprocessing - Filtering tokens and lemmatizing...")
    lemmatized_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            lemmatized_tokens.append(token.lemma_)

    print(f"INFO: Preprocessing complete. Found {len(sentences)} sentences, {len(lemmatized_tokens)} lemmatized tokens.")
    return cleaned_text, doc, sentences, lemmatized_tokens


# Add type hint for spacy Doc if not already implicitly covered by 'import spacy'
# from spacy.tokens import Doc # Optional explicit import for type hint

def extract_concepts(doc: spacy.tokens.Doc, cleaned_text: str, lemmatized_tokens: list) -> dict:
    """
    Extracts concepts using NER, RAKE, and TF.

    Args:
        doc: The processed spaCy Doc object from preprocess_text.
        cleaned_text: Lowercased, stripped text.
        lemmatized_tokens: List of filtered, lemmatized tokens.

    Returns:
        A dictionary containing lists of concepts:
        {'ner': ner_list, 'rake': rake_list, 'tf': tf_list}
    """
    concepts = {'ner': [], 'rake': [], 'tf': []}
    # Using print statements for logging/status
    print("INFO: Extracting Concepts...")

    # 1. NER Concepts
    print("INFO:  - Extracting NER entities...")
    try:
        # Using the 'doc' object passed as argument
        if doc is None:
             print("ERROR: Invalid spaCy Doc object received for NER.")
             concepts['ner'] = [] # Ensure key exists
        else:
            relevant_labels = ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'FAC', 'LOC', 'NORP']
            entities = []
            seen_entities = set()
            for ent in doc.ents:
                if ent.label_ in relevant_labels:
                    entity_text = ent.text.strip()
                    # Check if not already seen and has minimal length
                    if entity_text.lower() not in seen_entities and len(entity_text) > 1:
                         entities.append({'text': entity_text, 'label': ent.label_})
                         seen_entities.add(entity_text.lower())
            concepts['ner'] = entities
            print(f"INFO:    Found {len(entities)} unique relevant NER entities.")
    except Exception as e:
        print(f"ERROR: An error occurred during NER extraction: {e}")
        concepts['ner'] = [] # Ensure key exists

    # 2. RAKE Keyphrases
    print("INFO:  - Extracting RAKE keyphrases...")
    try:
        global rake_nltk_var # Use the globally initialized Rake object
        if rake_nltk_var and cleaned_text:
            rake_nltk_var.extract_keywords_from_text(cleaned_text)
            ranked_phrases = rake_nltk_var.get_ranked_phrases_with_scores()
            # Filter based on score and length
            # Keep phrases with score > 1.0 (default min score basically)
            # Limit phrase length (e.g., max 4 words)
            rake_keyphrases = [phrase for score, phrase in ranked_phrases if score > 1.5 and 1 < len(phrase.split()) <= 4]
            concepts['rake'] = rake_keyphrases[:15] # Limit to top 15 meeting criteria
            print(f"INFO:    Found {len(rake_keyphrases)} RAKE keyphrases (filtered). Added top {len(concepts['rake'])}.")
        elif not rake_nltk_var:
             print("WARN: RAKE not initialized, skipping RAKE extraction.")
             concepts['rake'] = []
        else:
             print("WARN: No cleaned text provided for RAKE.")
             concepts['rake'] = []
    except Exception as e:
        print(f"ERROR: An error occurred during RAKE extraction: {e}")
        concepts['rake'] = []


    # 3. TF Keywords
    print("INFO:  - Extracting TF keywords...")
    try:
        if lemmatized_tokens:
            lemma_counts = Counter(lemmatized_tokens)
            top_n = 20
            most_common_lemmas = lemma_counts.most_common(top_n)
            # Store as simple list of terms or dict with freq? Dict is more informative.
            tf_keywords = [{'term': lemma, 'freq': freq} for lemma, freq in most_common_lemmas]
            concepts['tf'] = tf_keywords
            print(f"INFO:    Found {len(tf_keywords)} top TF keywords.")
        else:
            print("WARN: No lemmatized tokens provided, skipping TF.")
            concepts['tf'] = []
    except Exception as e:
        print(f"ERROR: An error occurred during TF extraction: {e}")
        concepts['tf'] = []

    print("INFO: Concept extraction complete.")
    return concepts



def summarize_text(text_to_summarize: str, min_len: int = 150, max_len: int = 300) -> str | None:
    """
    Generates a summary for the input text using a transformer pipeline.

    Args:
        text_to_summarize: The text content to summarize (e.g., cleaned_text).
        min_len: Minimum length of the summary tokens.
        max_len: Maximum length of the summary tokens.

    Returns:
        The generated summary string, or None if summarization fails.
    """
    print("INFO: Summarizing text...")
    if not text_to_summarize or not isinstance(text_to_summarize, str):
        print("ERROR: Input text for summarization is empty or invalid.")
        return None

    # Attempt to initialize the pipeline (it will only load if None)
    summarizer = initialize_summarizer()
    if not summarizer:
        print("ERROR: Summarization failed - pipeline is not available.")
        return None

    try:
        input_length = len(text_to_summarize.split())
        # Adjust max_len based on input, ensuring it's greater than min_len
        # Avoid excessively large max_len for short inputs, prevent potential model errors
        adjusted_max_len = min(max_len, max(min_len + 20, int(input_length * 0.6))) # Reduced multiplier slightly
        adjusted_min_len = min(min_len, max(10, adjusted_max_len - 20)) # Ensure min_len isn't too large either

        print(f"INFO: Input length: ~{input_length} words. Target summary length: {adjusted_min_len}-{adjusted_max_len} tokens.")

        summary_result = summarizer(
            text_to_summarize,
            max_length=adjusted_max_len,
            min_length=adjusted_min_len,
            do_sample=False, # Keep deterministic output
            truncation=True  # Ensure input is truncated if too long for model
            )

        # Check if result is valid
        if summary_result and isinstance(summary_result, list) and 'summary_text' in summary_result[0]:
            generated_summary = summary_result[0]['summary_text']
            print("INFO: Summary generation complete.")
            return generated_summary
        else:
            print("ERROR: Summarization pipeline returned an unexpected result format.")
            return None

    except Exception as e:
        print(f"ERROR: An error occurred during summary generation: {e}")
        # Consider logging the error for more detail
        return None
    



# Add type hint for list if needed (from typing import list) - usually not required
def generate_questions(sentences: list, concepts: dict, num_questions: int = 15) -> list:
    """
    Generates questions based on lecture content using triggers and context.

    Args:
        sentences: List of spaCy sentence spans from preprocess_text.
        concepts: Dictionary of extracted concepts from extract_concepts.
        num_questions: Target number of unique questions to generate.

    Returns:
        A list of unique generated question strings.
    """
    print("INFO: Generating Questions...")
    if not sentences:
        print("ERROR: No sentences provided for question generation.")
        return []

    # --- Create trigger pool from concepts ---
    # (Keep this logic inside the function as it depends on the 'concepts' input)
    trigger_pool = set()
    # Add RAKE phrases if available
    if concepts.get('rake'):
        # Ensure items in 'rake' are strings
        trigger_pool.update([str(phrase).lower() for phrase in concepts['rake'] if isinstance(phrase, str)])
    # Add specific NER entities if available
    if concepts.get('ner'):
        # Ensure items in 'ner' are dicts with 'text' key
        trigger_pool.update([ent['text'].lower() for ent in concepts['ner'] if isinstance(ent, dict) and 'text' in ent and (len(ent['text'].split()) > 1 or ent.get('label') in ['ORG', 'PRODUCT', 'GPE'])])
    # Add TF keywords if needed and available
    if len(trigger_pool) < 20 and concepts.get('tf'):
         # Ensure items in 'tf' are dicts with 'term' key
         trigger_pool.update([term['term'] for term in concepts['tf'][2:15] if isinstance(term, dict) and 'term' in term and len(term['term']) > 3])

    trigger_pool = {term for term in trigger_pool if len(term) > 3} # Final length check
    print(f"INFO: Trigger pool size for this run: {len(trigger_pool)} items")
    if not trigger_pool:
        print("WARN: Trigger pool is empty. Cannot generate targeted questions.")
        return []

    # --- Find trigger-context pairs ---
    potential_pairs = []
    seen_contexts = set()
    shuffled_sentences = list(sentences)
    random.shuffle(shuffled_sentences)
    # Adjust num_contexts based on pool size? Maybe not needed.
    num_contexts_to_find = min(len(trigger_pool) * 2, num_questions * 3, len(sentences)) # Find reasonable number

    for sentence in shuffled_sentences:
        if len(potential_pairs) >= num_contexts_to_find: break
        # Ensure sentence is a spaCy Span and get text
        try:
            sentence_text = sentence.text.strip()
            sentence_lower = sentence_text.lower()
        except AttributeError:
            continue # Skip if not a valid sentence span object

        if 50 < len(sentence_text) < 600 and sentence_text not in seen_contexts:
             sorted_triggers = sorted(list(trigger_pool), key=len, reverse=True)
             for trigger_term in sorted_triggers:
                 # Use regex for potentially better matching
                 try:
                     if re.search(r'\b' + re.escape(trigger_term) + r'\b', sentence_lower):
                         match = re.search(re.escape(trigger_term), sentence_text, re.IGNORECASE)
                         original_casing_trigger = match.group(0) if match else trigger_term
                         potential_pairs.append({"answer": original_casing_trigger, "context": sentence_text})
                         seen_contexts.add(sentence_text)
                         break # Move to next sentence
                 except re.error:
                      # Handle potential regex errors if trigger_term is unusual
                      if trigger_term in sentence_lower: # Fallback to simple check
                            match = re.search(re.escape(trigger_term), sentence_text, re.IGNORECASE)
                            original_casing_trigger = match.group(0) if match else trigger_term
                            potential_pairs.append({"answer": original_casing_trigger, "context": sentence_text})
                            seen_contexts.add(sentence_text)
                            break # Move to next sentence

    print(f"INFO: Found {len(potential_pairs)} potential trigger-context pairs.")
    if not potential_pairs: return []

    # --- Initialize QG components ---
    if not initialize_qg_components(): # Call the initializer function
        print("ERROR: QG failed - Components not available.")
        return []
    # Access the globally loaded components (safer to check if None again)
    global qg_tokenizer, qg_model, qg_device
    if qg_tokenizer is None or qg_model is None or qg_device is None:
         print("ERROR: QG components not loaded correctly after initialization call.")
         return []

    # --- Generate Questions ---
    generated_questions_list = []
    max_q_length = 64 # Max length for the generated question tokens
    num_beams = 4 # Using beam search for potentially better quality
    print(f"INFO: Generating questions from {len(potential_pairs)} pairs...")

    for i, pair in enumerate(potential_pairs):
        answer_text = pair['answer'] # This is the trigger term
        context = pair['context']
        # Print progress less frequently
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(potential_pairs):
            print(f"INFO:   Processing QG pair {i+1}/{len(potential_pairs)}...")

        # Format input based on model expectations
        input_text = f"answer: {answer_text} context: {context}"

        try:
            # Tokenize input
            inputs = qg_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(qg_device)
            # Generate output (question)
            outputs = qg_model.generate(
                inputs["input_ids"],
                max_length=max_q_length,
                num_beams=num_beams,
                early_stopping=True
            )
            # Decode the generated tokens
            generated_text = qg_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # --- Extract and Validate the Question Part ---
            question_mark_index = generated_text.find('?')
            if question_mark_index != -1:
                potential_question = generated_text[:question_mark_index + 1].strip()
                # Basic validation: check length
                if len(potential_question.split()) > 3:
                    generated_questions_list.append(potential_question)
            # else: # Optional: log if no '?' was found
            #     print(f"WARN: No '?' found in QG output: {generated_text}")
            # --- End Validation ---

        except Exception as e:
            # Log errors infrequently to avoid flooding console
            if i % 10 == 0:
                 print(f"WARN: Error during QG for trigger '{answer_text}': {e}")
            continue # Continue to next pair even if one fails

    # --- De-duplicate and limit results ---
    unique_questions = list(dict.fromkeys(generated_questions_list))
    final_questions = unique_questions[:num_questions] # Limit to the desired number

    print(f"INFO: Question generation complete. Generated {len(final_questions)} unique questions.")
    return final_questions