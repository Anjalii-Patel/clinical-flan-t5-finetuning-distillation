"""
Interactive Gradio Web Application for Clinical Consultation Extraction.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import CLINICAL_QUESTIONS, ClinicalQAPipeline

DEFAULT_CONSULTATION = """What is your name? My name is Selena Smith. What brings you in today? I've been experiencing lower back pain for about three weeks. It started as a dull ache but has recently become sharp, especially when I bend or sit for long periods. Did anything specific trigger the pain, like an injury or heavy lifting? No, it came on gradually, and I can't recall any specific incident. Do you have any other symptoms, such as tingling, numbness, or weakness? Yes, occasionally, I feel a tingling sensation that runs down my right leg. Have you tried any treatments or medications? I've taken over-the-counter painkillers, but they don't seem to help much. Do you have any past medical history or chronic conditions? I sometimes get migraines and take ibuprofen for them, but nothing major otherwise. Have you been hospitalized or had any surgeries? Yes, I was hospitalized three years ago for a minor surgery to remove a cyst. What about your OB/GYN history? Nothing significant, apart from two normal deliveries about ten years ago. What is your lifestyle like? I work as a teacher, so I sit for long hours. I don't smoke or drink, and I try to walk daily, but I haven't been consistent because of the pain. Do you have any family history of similar issues or other conditions? My parents are generally healthy, though my father has mild arthritis. Are you aware of any allergies or hypersensitivities? I had a mild rash after taking amoxicillin years ago, but I'm not sure if it's an allergy. Have you undergone any previous investigations or tests related to this issue? No, not for my back, though I had a blood test last year, which came back normal. Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine."""


def create_demo(pipeline: ClinicalQAPipeline):
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is not installed. Run `pip install gradio` to run the web demo.")

    def run_full_extraction(context):
        if not context or not context.strip():
            context = DEFAULT_CONSULTATION
        res = pipeline.extract_full_consultation(context)
        meta = res.get("_meta", {})
        meta_str = f"Extracted 10 sections in {meta.get('total_latency_ms', 0):.1f} ms ({meta.get('avg_latency_per_question_ms', 0):.1f} ms/query)"
        return (
            res.get("brief_history", ""),
            res.get("chief_complaints", ""),
            res.get("past_medical_history", ""),
            res.get("hospitalization_surgical_history", ""),
            res.get("gynecological_history", ""),
            res.get("lifestyle_social_activity", ""),
            res.get("family_history", ""),
            res.get("allergies_hypersensitivities", ""),
            res.get("previous_investigations", ""),
            res.get("review_of_systems", ""),
            meta_str,
        )

    def run_single_question(question, context):
        if not context or not context.strip():
            context = DEFAULT_CONSULTATION
        res = pipeline.answer_question(question=question, context=context)
        return res["answer"], f"Latency: {res['latency_ms']:.1f} ms"

    with gr.Blocks(title="FLAN-T5 Clinical Consultation Extraction") as demo:
        gr.Markdown("# 🏥 FLAN-T5 Clinical Consultation Summarizer & QA")
        gr.Markdown(
            "Extract structured Electronic Health Record (EHR) categories from doctor-patient conversation transcripts."
        )

        with gr.Row():
            consultation_input = gr.Textbox(
                label="Doctor-Patient Consultation Transcript",
                value=DEFAULT_CONSULTATION,
                lines=8,
                placeholder="Paste consultation dialogue here...",
            )

        with gr.Row():
            extract_all_btn = gr.Button("📋 Extract Full Consultation Sections", variant="primary")

        latency_output = gr.Markdown("")

        gr.Markdown("### Structured Clinical Sections")
        with gr.Row():
            brief_out = gr.Textbox(label="1. Brief History", lines=2)
            complaints_out = gr.Textbox(label="2. Chief Complaints", lines=2)

        with gr.Row():
            past_out = gr.Textbox(label="3. Past Medical History", lines=2)
            surgery_out = gr.Textbox(label="4. Hospitalization & Surgical History", lines=2)

        with gr.Row():
            gyn_out = gr.Textbox(label="5. Gynecological History", lines=2)
            lifestyle_out = gr.Textbox(label="6. Lifestyle & Social Activity", lines=2)

        with gr.Row():
            family_out = gr.Textbox(label="7. Family History", lines=2)
            allergy_out = gr.Textbox(label="8. Allergies & Hypersensitivities", lines=2)

        with gr.Row():
            investigation_out = gr.Textbox(label="9. Previous Investigations", lines=2)
            ros_out = gr.Textbox(label="10. Review of Systems", lines=2)

        extract_all_btn.click(
            run_full_extraction,
            inputs=[consultation_input],
            outputs=[
                brief_out,
                complaints_out,
                past_out,
                surgery_out,
                gyn_out,
                lifestyle_out,
                family_out,
                allergy_out,
                investigation_out,
                ros_out,
                latency_output,
            ],
        )

        gr.Markdown("---")
        gr.Markdown("### Custom Clinical Query")
        with gr.Row():
            custom_q = gr.Textbox(label="Ask a custom question about this transcript", lines=1, placeholder="e.g. Has the patient taken any pain relievers?")
            custom_ans = gr.Textbox(label="Answer", lines=2)
        custom_latency = gr.Markdown("")
        ask_btn = gr.Button("Ask Question")
        ask_btn.click(
            run_single_question,
            inputs=[custom_q, consultation_input],
            outputs=[custom_ans, custom_latency],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio app for clinical extraction.")
    parser.add_argument("--model_path", type=str, default="google/flan-t5-base", help="Model checkpoint path or HF ID.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link.")
    args = parser.parse_args()

    pipeline = ClinicalQAPipeline(model_name_or_path=args.model_path)
    demo = create_demo(pipeline)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)

