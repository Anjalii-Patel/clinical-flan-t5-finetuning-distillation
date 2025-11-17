import gradio as gr

import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the FLAN-T5 model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def answer_question(context, question):
    # Preprocess the input text
    input_text = f"question: {question} context: {context}"
    # print(context)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate answer
    with torch.no_grad():
        answer_ids = model.generate(input_ids, max_length=200, num_beams=5, early_stopping=True)

    # Decode the answer
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return answer

input_text = "What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can’t recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I’ve taken over-the-counter painkillers, but they don’t seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don’t smoke or drink, and I try to walk daily, but I haven’t been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I’m not sure if it’s an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine."
# start_time = time.time()  # Record the start time
# output = answer_question(model, tokenizer)
# end_time = time.time()  # Record the end time

# Calculate elapsed time
# elapsed_time = end_time - start_time
# print(f"Time consumed by the model: {elapsed_time:.4f} seconds")

def process_input(data=input_text):
    question_array = ["What is brief history?","What are chief complaints?","What is past medical history?",
                      "What is Hospitalization & Surgical History?","What is Patient's Gynec History?",
                      "What is patient's Lifestyle & Social activity?","What is Patient's Family history?",
                      "What are Patien's Allergies & Hypersensitivity?","What is Patient's Previous Investigation?",
                      "What is overall Review of system?"]


    output_answers = []
    for i, question in enumerate(question_array):
        # if i>=3:
        #     output_answers.append("Temporary answer")
        #     continue
        ans = answer_question(data, question)
        output_answers.append(ans)
        print(f"Question:",question,"\nAnswer:", ans)
    return output_answers

with gr.Blocks() as demo:
    transcription = gr.Textbox(label="Transcription", value=input_text, lines=5, interactive=False)

    submit_button = gr.Button("Submit")

    brief = gr.Textbox(value="", label="Brief History", interactive=False)
    chief_complaints = gr.Textbox(value="", label="Chief complaints", interactive=False)
    past_history = gr.Textbox(value="", label="Past Medical History", interactive=False)
    hosp_surgihistory = gr.Textbox(value="", label="Hospitalization & Surgical History", interactive=False)
    gyn_history = gr.Textbox(value="", label="OB/GYN History", interactive=False)
    life_social = gr.Textbox(value="", label="Lifestyle & Social", interactive=False)
    family = gr.Textbox(value="", label="Family", interactive=False)
    allergy = gr.Textbox(value="", label="Allergies & Hypersensitivity", interactive=False)
    previous = gr.Textbox(value="", label="Previous Investigation", interactive=False)
    review = gr.Textbox(value="", label="Review of system", interactive=False)


    # Link the button to the function
    submit_button.click(process_input, inputs=None, outputs=[brief,chief_complaints,past_history,hosp_surgihistory,gyn_history,life_social,family,allergy,previous,review])

# Launch the interface
demo.launch(debug=True)
