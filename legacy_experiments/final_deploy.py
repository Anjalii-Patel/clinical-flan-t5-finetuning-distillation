import gradio as gr
from tqdm import tqdm
import numpy as np
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import scipy.io.wavfile as wavfile
import time
from datetime import datetime
# from accelerate import infer_auto_device_map
import transformers
# from tqdm import tqdm





full_consultation_qa_list = [
    "What is a brief history of patient?",
    "What are chief complaints of patient?",
    "What is past medical history of patient?",
    "What is Hospitalization & Surgical History of patient?",
    "What is a Patient's Gynec History?",
    "What is a patient's Lifestyle & Social activity?",
    "What is the Patient's Family history?",
    "What are Patient's Allergies & Hypersensitivity?",
    "What is Patient's Previous Investigation?",
    "What is the overall Review of the system of patient?",

]

default_data = """
What is your name? My name is Selena Smith. What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can’t recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I’ve taken over-the-counter painkillers, but they don’t seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don’t smoke or drink, and I try to walk daily, but I haven’t been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I’m not sure if it’s an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine.
"""

# Load Whisper Large model and processor for transcription
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
transcription_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
transcription_model.eval()
# transcription_model.config.forced_decoder_ids = None



# Load Flan-T5 for QA
model_name = "BioMistral/BioMistral-7B-Dare"
# model_name = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMV"

# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=False)

tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    # max_memory={0: "2GiB", 1: "3GiB", 2: "3GiB", "cpu": "64GiB"},
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
                                                model_name,
                                                # load_in_4bit=True,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto",
                                                max_memory={0: "2GiB", 1: "3GiB", 2: "3GiB", "cpu": "64GiB"},
                                                # attn_implementation="flash_attention_2"
                                                quantization_config=bnb_config
                                             )

# device_map = infer_auto_device_map(
#     model,
#     max_memory={0: "4GiB", 1: "4GiB", 2: "4GiB", "cpu": "32GiB"}
# )

# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              load_in_4bit=True,
#                                             #  torch_dtype=torch.float16,
#                                              device_map=device_map
#                                             #  quantization_config=quantization_config
#                                              )

# model.half()
model = torch.compile(model)
model.eval()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.Client(
    Settings(
    #           chroma_db_impl="duckdb+parquet",
            persist_directory="./db/"
            )
                                  )

collection = client.create_collection(name="Medical_Transcripts",embedding_function=embedding_fn)

# Load T5 model for summarization
summarizer = pipeline("summarization", model="t5-large")  # Use an appropriate model for summarization

# Use CPU for inference
device_name = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
print("Computation selection:", device_name)
device = torch.device(device_name)
# transcription_model = torch.nn.DataParallel(transcription_model)
transcription_model.to(device)


# Function to transcribe audio
def transcribe_audio(audio, language="English"):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="translate")
    if audio is None:
        return "No audio provided."

    # Extract the sample rate and audio array from the Gradio input
    sr, y = audio

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Normalize the audio data
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # Resample the audio to 16kHz (Whisper model's required sampling rate)
    if sr != 16000:
        y_tensor = torch.tensor(y)
        resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        y = resample_transform(y_tensor).numpy()
        sr = 16000

    # Save audio data for the archival process
    datetime_stamp = datetime_stamp = str(datetime.now()).replace("-", "_").replace(" ", "_").replace(":", "_").replace(".", "_")
    file_name = f"gradio_output/recording_{datetime_stamp}.wav"
    wavfile.write(file_name, sr, (y * 32767).astype(np.int16))

    # Segment the audio into 15-second chunks
    segment_length = 15 * sr  # 15 seconds in samples
    num_segments = len(y) // segment_length
    segments = [y[i * segment_length: (i + 1) * segment_length] for i in range(num_segments)]

    # Add the last segment if there's any remaining audio data
    if len(y) % segment_length != 0:
        segments.append(y[num_segments * segment_length:])

    transcriptions = []

    # Process each segment
    for segment in tqdm(segments):
        with torch.no_grad():
            input_features = processor(segment, sampling_rate=sr, return_tensors="pt").input_features.to(device)

        predicted_ids = transcription_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()
        transcriptions.append(transcription)

    # Join the transcriptions of each segment
    full_transcription = " ".join(transcriptions)

    # Save the transcription for the archival process
    transcription_file = file_name.replace(".wav", ".txt")
    with open(transcription_file, "w") as f:
        f.write(full_transcription)
        f.close()

    return full_transcription

# Function to perform QA with Flan-T5
def perform_qa(question, transcription=default_data):
    if not transcription.strip() or transcription.strip() == "":
        # transcription = default_data
        transcription= "What is your name? My name is Selena Smith. What brings you in today? I’ve been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can’t recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I’ve taken over-the-counter painkillers, but they don’t seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don’t smoke or drink, and I try to walk daily, but I haven’t been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I’m not sure if it’s an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine."

    collection.add(
        documents = [transcription],
        metadatas = [{"source": "transcription"}],
        ids = ["id1"]
        )
    results = collection.query(query_texts=[question],
                               n_results=1
                               )
    if results["documents"]:
        context = results["documents"][0][0]
    else:
        return "No relevant context found in the database."

    # qa_pairs = [
    #     {"context": context, "question": i} for i in question
    # ]

    # print("Debug context:", context)

    # input_text = f"give me the answer of the medical question based on only the context i am giving, do not answer out of the context if not asked. context: {context} now this is question: {question} "
    input_text = f"Answer the medical question strictly based on the given context. Do not use information beyond the context unless explicitly asked. Context: {context} Question: {question}"

    # # input_text = f"""
    # # You are a context-restricted assistant. Your primary goal is to answer questions based only on the provided context.
    # # If the required information is not found within the context, respond exactly as:
    # # "The required information is not available in the provided context."

    # # Do not infer, fabricate, or assume any information.
    # # this is context:{context}
    # # this is question:{question}
    # # Answer:
    # # """
    # # print("Debug Log:", input_text)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(torch.device("cuda"))

    with torch.inference_mode():
        answer_ids = model.generate(input_ids, num_beams=4, early_stopping=True,  max_new_tokens=150, do_sample=False)

    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    # print("Answer Debug:", answer)
    answer = answer.split(input_text)
    if len(answer) >=2:
        answer = answer[1]
    else:
        answer = answer[0]

    # Split QA pairs into batches
    # batch_size = 2
    # batches = [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)]
    # tokenizer.pad_token = tokenizer.eos_token
    # return_outputs = []

    # # Process each batch
    # for batch in tqdm(batches):
    #     formatted_inputs = [
    #         f"Answer the medical question strictly based on the given context.\nContext: {pair['context']}\nQuestion: {pair['question']}"
    #         for pair in batch
    #     ]
    #     batch_inputs = tokenizer(
    #         formatted_inputs,

    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=512
    #     ).to("cuda")

    #     with torch.no_grad():
    #         outputs = model.generate(**batch_inputs, num_beams=4, early_stopping=True,  max_new_tokens=150, do_sample=False)

    #     responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    #     edited_responses = []
    #     for i in responses:
    #         temp = i.split("Answer:")[1].strip()
    #         edited_responses.append(temp)
    #     return_outputs.extend(edited_responses)

        # Print results
        # for pair, response in zip(batch, responses):
            # print(f"Context: {pair['context']}")
            # print(f"Question: {pair['question']}")
            # print(f"Answer: {response}")
            # print("-" * 50)

    return answer

# Function to generate a summary using T5
def generate_summary(transcription):
    if not transcription.strip():
        transcription = default_data
        # return "No transcription available for summary."

    # Use the T5 model to summarize the transcription
    summary = summarizer(transcription, max_length=150, min_length=50, do_sample=False, num_beams=5)
    return "Summary: " + summary[0]['summary_text']

def auto_full_qa(context):
    if not context or context.strip() == "":
        context = default_data

    outputs = [""] * len(full_consultation_qa_list)
    # start = time.time()
    # outputs = perform_qa(full_consultation_qa_list, context)
    # print("Time taken:", int(time.time() - start))
    # return outputs

    for i in range(len(full_consultation_qa_list)):
        start = time.time()
        outputs[i] = perform_qa(full_consultation_qa_list[i], context) #"Temp answer"
        print("Time taken:", int(time.time() - start))
        print(f"Answer[{i}]:", outputs[i])
        yield outputs

# Gradio interface
with gr.Blocks() as interface:
    with gr.Row():
        gr.Markdown("# Speech AI for MedicalQA")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
        language_button = gr.Radio(["English", "Hindi"], label="Select the language")#, "Gujarati", "Marathi", "Punjabi"], label="Select the language")

    with gr.Row():
        transcription_output = gr.Textbox(
            label="Transcription",
            placeholder=default_data,
            lines=10,
            interactive=True,
        )

    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a Question",
            placeholder="Enter your question here...",
            lines=2,
        )
        answer_output = gr.Textbox(
            label="Answer",
            placeholder="The answer will be here...",
            lines=5,
            interactive=False
        )

    with gr.Row():
        submit_button = gr.Button("Submit")
        summary_button = gr.Button("Summary")
        auto_qa = gr.Button("Full Consultation")
        qa_button = gr.Button("Ask Question")

    with gr.Row():
        summary_output = gr.Textbox(
            label="Summary",
            placeholder="Your summary will be here...",
            lines=5,
            interactive=False
        )
    with gr.Row():
        gr.Markdown("# Default QA available on BMC")

    full_qa_object_list = list()

    for i in range(len(full_consultation_qa_list)):

        with gr.Row():
            temp = gr.Textbox(
            label= f"Q{i+1} : {full_consultation_qa_list[i]}",
            lines=2,
        )
            full_qa_object_list.append(temp)

    # Connect the components
    submit_button.click(transcribe_audio, inputs=[audio_input, language_button], outputs=transcription_output)
    summary_button.click(generate_summary, inputs=transcription_output, outputs=summary_output)
    qa_button.click(perform_qa, inputs=[question_input, transcription_output], outputs=answer_output)
    auto_qa.click(auto_full_qa, inputs=transcription_output, outputs=full_qa_object_list)

# Launch the Gradio app
interface.launch(
    server_name="172.16.16.25", # server ip
    server_port=7860, # server port
    ssl_verify=False, # ssl verification, switch to True when certificate is available
    ssl_certfile="cert.pem", # ssl demp certificate
    ssl_keyfile="key.pem" # ssl demo keyfile
)
