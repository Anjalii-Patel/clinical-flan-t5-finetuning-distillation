import warnings
warnings.filterwarnings("ignore")


# # from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
# # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# # input = tokenizer("",return_tensors="pt")

# # answer = model.generate(input,max_length=300,temperature=0.7,top_p=0.9)
# # output = tokenizer.batch_decode(answer, skip_special_tokens=True)
# # print("Answer:",output)


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "google/flan-t5-large"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# def answer_question(question: str, context: str = None) -> str:
#     # Combine question and context
#     input_text = f"Question: {question}"
#     if context:
#         input_text += f" Context: {context}"

#     # Tokenize input
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

#     # Generate answer
#     outputs = model.generate(**inputs, max_length=448, num_beams=5, early_stopping=True)

#     # Decode and return the answer
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Example: QA without additional context
# question = "What are chief complaints?"
# answer = answer_question(question)
# # print(f"Q: {question}\nA: {answer}")

# # Example: QA with context
# context = "A patient visits the doctor, reporting fatigue, headaches, irritability, and trouble concentrating, possibly due to stress. The doctor inquires about the symptoms' duration, sleep, and diet changes, learning the issues started three weeks ago, with irregular sleep and occasional nausea. The doctor suggests blood tests to rule out other causes like anemia or hormonal imbalances and advises a consistent sleep schedule, balanced meals, and light exercise to manage stress. The patient agrees to follow the advice and schedule tests."
# answer_with_context = answer_question(question, context)
# print(f"\nQ: {question}\nContext: {context}\nA: {answer_with_context}")


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the FLAN-T5 model and tokenizer
model_name = "C:/Anjali/Fine Tuning Flan-T5/fine_tuned_flan_t5_large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Function to summarize text
# def summarize_text(text):
#     # Preprocess the input text
#     input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

#     # Generate summary
#     with torch.no_grad():
#         summary_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

#     # Decode the summary
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# Function to perform question answering
def answer_question(context, question):
    # Preprocess the input text
    input_text = f"context: {context} question: {question} "
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate answer
    with torch.no_grad():
        answer_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

    # Decode the answer
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return answer

input_text = "What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can’t recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I’ve taken over-the-counter painkillers, but they don’t seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don’t smoke or drink, and I try to walk daily, but I haven’t been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I’m not sure if it’s an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine."
question1 = "What is brief history?"
question2 = "What are chief complaints?"
question3 = "What is past medical history?"
question4 = "What is Hospitalization & Surgical History?"
question5 = "What is Patient's Gynec History?"
question6 = "What is patient's Lifestyle & Social activity?"
question7 = "What is Patient's Family history?"
question8 = "What are Patien's Allergies & Hypersensitivity?"
question9 = "What is Patient's Previous Investigation?"
question10 = "What is overall Review of system?"

# # Summarize the input text
# summary = summarize_text(input_text)
# print("Summary:", summary)

# Perform question answering using the summarized text
answer1 = answer_question(input_text, question1)
answer2 = answer_question(input_text, question2)
answer3 = answer_question(input_text, question3)
answer4 = answer_question(input_text, question4)
answer5 = answer_question(input_text, question5)
answer6 = answer_question(input_text, question6)
answer7 = answer_question(input_text, question7)
answer8 = answer_question(input_text, question8)
answer9 = answer_question(input_text, question9)
answer10 = answer_question(input_text, question10)


print("Question1:",question1,"Answer1:", answer1)
print("Question2:",question2,"Answer2:", answer2)
print("Question3:",question3,"Answer3:", answer3)
print("Question4:",question4,"Answer4:", answer4)
print("Question5:",question5,"Answer5:", answer5)
print("Question6:",question6,"Answer6:", answer6)
print("Question7:",question7,"Answer7:", answer7)
print("Question8:",question8,"Answer8:", answer8)
print("Question9:",question9,"Answer9:", answer9)
print("Question10:",question10,"Answer10:", answer10)

output_answers = [answer1,answer2,answer3,answer4,answer5,answer6,answer7,answer8,answer9,answer10]