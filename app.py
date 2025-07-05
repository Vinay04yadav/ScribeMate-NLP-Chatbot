# app.py - Main Streamlit application file for ScribeMate Chat

import streamlit as st
import time # Keep for potential future use (e.g., simulated typing)

# --- Import NLP functions ---
# Make sure nlp_processor.py is in the same directory and correct
try:
    from nlp_processor import (
        preprocess_text,
        extract_concepts,
        summarize_text,
        generate_questions
        # Initializer functions are called within the main functions in nlp_processor.py
    )
    NLP_FUNCTIONS_LOADED = True
    print("INFO: NLP functions loaded successfully from nlp_processor.py")
except ImportError:
    # Use st.error for UI feedback if Streamlit has started rendering
    st.error("FATAL ERROR: Failed to import functions from nlp_processor.py. Make sure the file exists and has no syntax errors.")
    print("FATAL ERROR: Failed to import functions from nlp_processor.py.")
    NLP_FUNCTIONS_LOADED = False
    st.stop() # Stop the script if core functions can't be loaded
except Exception as e:
    st.error(f"FATAL ERROR: An unexpected error occurred during NLP function import: {e}")
    print(f"FATAL ERROR: An unexpected error occurred during NLP function import: {e}")
    NLP_FUNCTIONS_LOADED = False
    st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="ScribeMate Chat", layout="centered", initial_sidebar_state="collapsed")
st.title("🤖 ScribeMate Chat")
print("--- DEBUG: app.py execution started ---") # First debug print


# --- Reset Function ---
def reset_chat():
    """Resets all relevant session state variables for a new conversation."""
    print("--- DEBUG: Resetting chat state ---")
    st.session_state.messages = [{"role": "assistant", "content": "Okay, starting over! What would you like to do: 'summary' or 'questions'?"}]
    st.session_state.processing_stage = "AWAITING_INTENT"
    st.session_state.transcript = None
    st.session_state.cleaned_text = None
    st.session_state.doc = None
    st.session_state.sentences = None
    st.session_state.lemmatized_tokens = None
    st.session_state.concepts = None
    st.session_state.summary = None
    st.session_state.questions = None
    st.session_state.base_processing_done = False


# --- Initialize Session State Variables ---
def initialize_state():
    """Initializes session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm ScribeMate. I can help summarize or generate study questions from lecture transcripts. What would you like to do today ('summary' or 'questions')?"}]
    if "processing_stage" not in st.session_state:
        st.session_state.processing_stage = "AWAITING_INTENT"
    if "transcript" not in st.session_state: st.session_state.transcript = None
    if "cleaned_text" not in st.session_state: st.session_state.cleaned_text = None
    if "doc" not in st.session_state: st.session_state.doc = None
    if "sentences" not in st.session_state: st.session_state.sentences = None
    if "lemmatized_tokens" not in st.session_state: st.session_state.lemmatized_tokens = None
    if "concepts" not in st.session_state: st.session_state.concepts = None
    if "summary" not in st.session_state: st.session_state.summary = None
    if "questions" not in st.session_state: st.session_state.questions = None
    if "base_processing_done" not in st.session_state: st.session_state.base_processing_done = False

print("--- DEBUG: Calling initialize_state ---")
initialize_state()
print(f"--- DEBUG: Initial stage = {st.session_state.processing_stage} ---")


# --- Display Existing Chat Messages ---
print("--- DEBUG: Displaying chat history ---")
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Ensure content is a string before passing to markdown
        content_to_display = str(message.get("content", ""))
        st.markdown(content_to_display)
print("--- DEBUG: Finished displaying history ---")


# --- Helper function for base processing ---
# Performs preprocessing and concept extraction, stores results in session state
def perform_base_processing():
    """Runs preprocessing and concept extraction, returns True on success."""
    if st.session_state.transcript and not st.session_state.base_processing_done:
         print("--- DEBUG: Starting base processing ---")
         preprocessing_result = preprocess_text(st.session_state.transcript)
         if preprocessing_result:
             st.session_state.cleaned_text, st.session_state.doc, st.session_state.sentences, st.session_state.lemmatized_tokens = preprocessing_result
             st.session_state.concepts = extract_concepts(st.session_state.doc, st.session_state.cleaned_text, st.session_state.lemmatized_tokens)
             st.session_state.base_processing_done = True
             print("--- DEBUG: Base processing complete ---")
             return True
         else:
             print("--- DEBUG: Preprocessing failed ---")
             st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't preprocess the transcript. Please check the text or try 'reset'."})
             st.session_state.processing_stage = "ERROR" # Mark error state
             return False
    elif st.session_state.base_processing_done:
         print("--- DEBUG: Base processing already done ---")
         return True # Already done, success
    else:
         print("--- DEBUG: No transcript found for base processing ---")
         st.session_state.messages.append({"role": "assistant", "content": "Something went wrong - no transcript found for processing. Please type 'reset'."})
         st.session_state.processing_stage = "ERROR"
         return False


# --- Handle User Input ---
print("--- DEBUG: Setting up chat input ---")
if prompt := st.chat_input("What would you like to do, or paste transcript..."):
    print(f"--- DEBUG: User prompt received: '{prompt[:50]}...' ---")
    # Add user's message to history (display happens on next rerun)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Universal Reset Check ---
    user_input_lower = prompt.lower().strip()
    if "new transcript" in user_input_lower or "start over" in user_input_lower or "reset" in user_input_lower:
        print("--- DEBUG: Reset command detected ---")
        reset_chat()
        st.rerun() # Rerun immediately after reset to show initial prompt

    # --- Stage-Specific Logic (only if not reset) ---
    else:
        current_stage = st.session_state.processing_stage
        print(f"--- DEBUG: Handling input for stage: {current_stage} ---")

        # 1. Stage: Awaiting Intent
        if current_stage == "AWAITING_INTENT":
            if "summary" in user_input_lower:
                st.session_state.processing_stage = "AWAITING_TRANSCRIPT_FOR_SUMMARY"
                st.session_state.messages.append({"role": "assistant", "content": "Great! Please paste the full lecture transcript you want summarized."})
            elif "question" in user_input_lower:
                st.session_state.processing_stage = "AWAITING_TRANSCRIPT_FOR_QUESTIONS"
                st.session_state.messages.append({"role": "assistant", "content": "Okay! Please paste the full lecture transcript you want questions generated from."})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I didn't catch that. Do you want a 'summary' or 'questions' generated? You can also type 'reset'."})
            st.rerun()

        # 2. Stage: Awaiting Transcript for Summary
        elif current_stage == "AWAITING_TRANSCRIPT_FOR_SUMMARY":
            if len(prompt) > 100: # Basic check if input looks like a transcript
                st.session_state.transcript = prompt
                st.session_state.processing_stage = "PROCESSING_SUMMARY"
                st.session_state.messages.append({"role": "assistant", "content": "Thanks! Generating the summary now..."})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "That seems a bit short for a transcript. Please paste the full text, or type 'reset'."})
            st.rerun()

        # 3. Stage: Awaiting Transcript for Questions
        elif current_stage == "AWAITING_TRANSCRIPT_FOR_QUESTIONS":
            if len(prompt) > 100:
                st.session_state.transcript = prompt
                st.session_state.processing_stage = "PROCESSING_QUESTIONS"
                st.session_state.messages.append({"role": "assistant", "content": "Got it! Generating questions now..."})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "That seems a bit short for a transcript. Please paste the full text, or type 'reset'."})
            st.rerun()

        # 4. Stage: After Displaying (Awaiting Follow-up)
        elif current_stage in ["DISPLAY_SUMMARY", "DISPLAY_QUESTIONS"]:
             follow_up_processed = False
             if "summary" in user_input_lower and current_stage == "DISPLAY_QUESTIONS":
                 if st.session_state.summary: # If already generated
                     st.session_state.processing_stage = "DISPLAY_SUMMARY"
                 else: # Need to generate it
                     st.session_state.processing_stage = "PROCESSING_SUMMARY"
                     st.session_state.messages.append({"role": "assistant", "content": "Okay, generating the summary for the same transcript..."})
                 follow_up_processed = True
                 st.rerun()
             elif "question" in user_input_lower and current_stage == "DISPLAY_SUMMARY":
                 if st.session_state.questions: # If already generated
                     st.session_state.processing_stage = "DISPLAY_QUESTIONS"
                 else: # Need to generate them
                     st.session_state.processing_stage = "PROCESSING_QUESTIONS"
                     st.session_state.messages.append({"role": "assistant", "content": "Okay, generating questions for the same transcript..."})
                 follow_up_processed = True
                 st.rerun()

             if not follow_up_processed:
                 other_option = "questions" if current_stage == "DISPLAY_SUMMARY" else "summary"
                 st.session_state.messages.append({"role": "assistant", "content": f"You can ask for '{other_option}' for this transcript, or type 'reset'."})
                 st.rerun()
        elif current_stage == "ERROR":
              st.session_state.messages.append({"role": "assistant", "content": "There was an error previously. Please type 'reset' to start over."})
              st.rerun()


# --- Background Processing Blocks ---
# These run when the stage is set by the input handler above and st.rerun() is called

# Block for Processing Summary
if st.session_state.processing_stage == "PROCESSING_SUMMARY":
    print("--- DEBUG: Entering PROCESSING_SUMMARY block ---")
    if NLP_FUNCTIONS_LOADED:
        with st.spinner("Generating summary..."):
            if perform_base_processing(): # Ensure transcript is preprocessed
                try:
                    print("--- DEBUG: Calling summarize_text ---")
                    st.session_state.summary = summarize_text(st.session_state.cleaned_text)
                    print(f"--- DEBUG: Summary result: {'Generated' if st.session_state.summary else 'None'} ---")
                    st.session_state.processing_stage = "DISPLAY_SUMMARY" # Ready to display
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")
                    print(f"--- ERROR: Exception during summarization: {e} ---")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error occurred while generating the summary."})
                    st.session_state.processing_stage = "ERROR"
            # else: Base processing failed, message added, stage set to ERROR
        # Rerun needed to display result or error message
        print(f"--- DEBUG: Exiting PROCESSING_SUMMARY block, rerunning. New stage: {st.session_state.processing_stage} ---")
        st.rerun()

# Block for Processing Questions
if st.session_state.processing_stage == "PROCESSING_QUESTIONS":
    print("--- DEBUG: Entering PROCESSING_QUESTIONS block ---")
    if NLP_FUNCTIONS_LOADED:
        with st.spinner("Generating questions... This might take a bit longer..."):
            if perform_base_processing(): # Ensure transcript is preprocessed
                try:
                    print("--- DEBUG: Calling generate_questions ---")
                    st.session_state.questions = generate_questions(st.session_state.sentences, st.session_state.concepts, num_questions=10)
                    print(f"--- DEBUG: Question result: {'Generated questions' if st.session_state.questions else 'None/Empty'} ---")
                    st.session_state.processing_stage = "DISPLAY_QUESTIONS" # Ready to display
                except Exception as e:
                    st.error(f"An error occurred during question generation: {e}")
                    print(f"--- ERROR: Exception during question generation: {e} ---")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, an error occurred while generating questions."})
                    st.session_state.processing_stage = "ERROR"
            # else: Base processing failed, message added, stage set to ERROR
        # Rerun needed to display result or error message
        print(f"--- DEBUG: Exiting PROCESSING_QUESTIONS block, rerunning. New stage: {st.session_state.processing_stage} ---")
        st.rerun()


# --- Display Results Blocks ---
# These execute when the stage is DISPLAY_... after a rerun from processing block

print(f"--- DEBUG: Reached display checks. Current stage: {st.session_state.processing_stage} ---")

if st.session_state.processing_stage == "DISPLAY_SUMMARY":
    # Only add the message if it wasn't the last one added already
    if not st.session_state.messages or st.session_state.messages[-1].get("content", "").find("Okay, here's the summary:") == -1:
        print("--- DEBUG: Adding summary message to display ---")
        with st.chat_message("assistant"):
            if st.session_state.summary:
                content = f"Okay, here's the summary:\n\n---\n\n{st.session_state.summary}\n\n---\n\nWould you like the 'questions' for this transcript, or type 'reset' to start a new one?"
                st.markdown(content)
                st.session_state.messages.append({"role": "assistant", "content": content}) # Add to history AFTER displaying
            else:
                content = "Sorry, I couldn't generate a summary. Would you like to try generating 'questions'? Or type 'reset'."
                st.markdown(content)
                st.session_state.messages.append({"role": "assistant", "content": content})

elif st.session_state.processing_stage == "DISPLAY_QUESTIONS":
     if not st.session_state.messages or st.session_state.messages[-1].get("content", "").find("Okay, here are the generated questions:") == -1:
        print("--- DEBUG: Adding questions message to display ---")
        with st.chat_message("assistant"):
            if st.session_state.questions:
                q_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(st.session_state.questions)])
                content = f"Okay, here are the generated questions:\n\n---\n\n{q_list}\n\n---\n\nWould you like the 'summary' for this transcript, or type 'reset' to start a new one?"
                st.markdown(content)
                st.session_state.messages.append({"role": "assistant", "content": content}) # Add to history AFTER displaying
            else:
                 content = "Sorry, I couldn't generate any questions. Would you like to try the 'summary'? Or type 'reset'."
                 st.markdown(content)
                 st.session_state.messages.append({"role": "assistant", "content": content})

print("--- DEBUG: Reached end of app.py script for this run ---")