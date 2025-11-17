# FOR LOADING FLAN-T5-LARGE MULTIPLE TIMES


# from multiprocessing import Process, Queue
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# import time

# # Model initialization (for each process)
# def initialize_model(model_name="google/flan-t5-large"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return tokenizer, model

# # Function to process a single question
# def process_question(query, context, result_queue, process_id):
#     tokenizer, model = initialize_model()

#     if not context:
#         result_queue.put((process_id, "Please provide a transcription!", 0))
#         return

#     input_text = f"""
#     Context:
#     {context}

#     Question:
#     {query}

#     Answer:
#     """
#     start_time = time.time()  # Start timing the process
#     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

#     with torch.inference_mode():
#         outputs = model.generate(inputs, max_length=300, num_beams=5, early_stopping=True)

#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     end_time = time.time()  # End timing the process

#     elapsed_time = end_time - start_time
#     result_queue.put((process_id, answer, elapsed_time))  # Include elapsed time in the result

# # Main function to handle multiprocessing
# def auto_full_qa(context, qa_process_list):
#     processes = []
#     result_queue = Queue()  # Shared queue to store results

#     # Launch a process for each question
#     for i, question in enumerate(qa_process_list):
#         p = Process(target=process_question, args=(question, context, result_queue, i))
#         processes.append(p)
#         p.start()

#     # Collect results
#     results = [""] * len(qa_process_list)
#     times = [0] * len(qa_process_list)  # Store processing times
#     for _ in range(len(qa_process_list)):
#         process_id, answer, elapsed_time = result_queue.get()  # Retrieve result from the queue
#         results[process_id] = answer
#         times[process_id] = elapsed_time

#     # Wait for all processes to finish
#     for p in processes:
#         p.join()

#     # Print time taken for each process
#     for i, elapsed_time in enumerate(times):
#         print(f"Process {i} took {elapsed_time:.2f} seconds.")

#     return results

# # Example usage
# if __name__ == "__main__":
#     context = "What is your name? My name is Selena Smith. What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can’t recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I’ve taken over-the-counter painkillers, but they don’t seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don’t smoke or drink, and I try to walk daily, but I haven’t been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I’m not sure if it’s an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine."
#     qa_process_list = [
#         "What is a brief history of patient?",
#         "What are chief complaints of patient?",
#         "What is past medical history of patient?",
#         "What is Hospitalization & Surgical History of patient?",
#         "What is a Patient's Gynec History?",
#         "What is a patient's Lifestyle & Social activity?",
#         "What is the Patient's Family history?",
#         "What are Patient's Allergies & Hypersensitivity?",
#         "What is Patient's Previous Investigation?",
#         "What is the overall Review of the system of patient?"
#     ]

#     start_time = time.time()
#     answers = auto_full_qa(context, qa_process_list)
#     print("Answers:", answers)
#     print("Total Time Taken:", time.time() - start_time)



# FOR LOADING FLAN-T5-LARGE SINGLE TIME
from multiprocessing import Process, Queue, Lock
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

def initialize_shared_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

def process_question(query, context, tokenizer, model, result_queue, process_id, lock):
    if not context:
        result_queue.put((process_id, "Please provide a transcription!", 0))
        return

    input_text = f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    start_time = time.time()

    with lock:  # Synchronize model access
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        with torch.inference_mode():
            outputs = model.generate(inputs, max_length=300, num_beams=5, early_stopping=True)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    end_time = time.time()
    elapsed_time = end_time - start_time

    result_queue.put((process_id, answer, elapsed_time))

def auto_full_qa(context, qa_process_list):
    tokenizer, model = initialize_shared_model()
    lock = Lock()  # Lock for synchronizing access to the model
    processes = []
    result_queue = Queue()

    for i, question in enumerate(qa_process_list):
        p = Process(target=process_question, args=(question, context, tokenizer, model, result_queue, i, lock))
        processes.append(p)
        p.start()

    results = [""] * len(qa_process_list)
    times = [0] * len(qa_process_list)
    for _ in range(len(qa_process_list)):
        process_id, answer, elapsed_time = result_queue.get()
        results[process_id] = answer
        times[process_id] = elapsed_time

    for p in processes:
        p.join()

    for i, elapsed_time in enumerate(times):
        print(f"Process {i} took {elapsed_time:.2f} seconds.")

    return results

# Example usage
if __name__ == "__main__":
    context = "What is your name? My name is Selena Smith. What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can’t recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I’ve taken over-the-counter painkillers, but they don’t seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don’t smoke or drink, and I try to walk daily, but I haven’t been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I’m not sure if it’s an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine."
    qa_process_list = [
        "What is a brief history of patient?",
        "What are chief complaints of patient?",
        "What is past medical history of patient?",
        "What is Hospitalization & Surgical History of patient?",
        "What is a Patient's Gynec History?",
        "What is a patient's Lifestyle & Social activity?",
        "What is the Patient's Family history?",
        "What are Patient's Allergies & Hypersensitivity?",
        "What is Patient's Previous Investigation?",
        "What is the overall Review of the system of patient?"]

    start_time = time.time()
    answers = auto_full_qa(context, qa_process_list)
    print("Answers:", answers)
    print("Total Time Taken:", time.time() - start_time)
