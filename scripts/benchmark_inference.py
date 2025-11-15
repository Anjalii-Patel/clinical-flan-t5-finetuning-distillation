"""
CLI entry point for benchmarking inference latency and extraction behavior on multi-turn consultations.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import CLINICAL_QUESTIONS, ClinicalQAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("benchmark")

# 10 clinical consultation dialogues preserved from original experimental benchmarking
BENCHMARK_CONTEXTS: List[str] = [
    "What is your name? My name is Daniel Thompson.What brings you in today? I’ve been having persistent chest discomfort for the past week. It’s not severe, but it feels like a tightness, especially when I exert myself.Did anything specific trigger the discomfort, like physical activity or stress? It seems to get worse when I climb stairs or carry heavy objects. I haven’t been under unusual stress lately.Do you have any other symptoms, such as shortness of breath, dizziness, or sweating? Yes, I sometimes feel a bit short of breath, but no dizziness or unusual sweating.Have you tried any treatments or medications? I’ve taken some antacids thinking it might be heartburn, but they didn’t really help.Do you have any past medical history or chronic conditions? I have high blood pressure, which I manage with medication, and I was diagnosed with high cholesterol two years ago.Have you been hospitalized or had any surgeries? Yes, I had my appendix removed when I was a teenager.What about your family medical history? My father had a heart attack in his early fifties, and my mother has type 2 diabetes.Are you aware of any allergies or hypersensitivities? I’m allergic to penicillin; it causes a rash.Have you undergone any previous investigations or tests related to this issue? No, this is the first time I’ve experienced something like this.Have you noticed any issues with other systems, such as your digestion, vision, or urinary habits? No, everything else seems normal.",
    "What is your name? My name is Sarah Johnson.What brings you in today? I've been experiencing severe headaches almost daily for the past month. They usually start in the morning and get worse throughout the day.Did anything specific trigger the headaches, like stress or lack of sleep? They seem to come on regardless of my sleep or stress levels.Do you have any other symptoms, such as nausea, visual changes, or sensitivity to light? Yes, I often feel nauseous, and bright lights make the headaches worse.Have you tried any treatments or medications? I've taken ibuprofen, but it doesn't provide much relief.Do you have any past medical history or chronic conditions? I have a history of seasonal allergies, but nothing major.Have you been hospitalized or had any surgeries? No, I haven't been hospitalized or had any surgeries.What about your family medical history? My mother suffers from migraines, and my father has high blood pressure.Are you aware of any allergies or hypersensitivities? I'm allergic to shellfish; it causes hives.Have you undergone any previous investigations or tests related to this issue? No, this is the first time I've sought medical advice for it.Have you noticed any issues with other systems, such as your digestion, heart, or lungs? No, everything else seems normal.",
    "What is your name? My name is Michael Rivera.What brings you in today? I've been having frequent episodes of dizziness and lightheadedness, especially when standing up quickly.Did anything specific trigger the dizziness, like dehydration or illness? Not that I can identify. It happens randomly throughout the day.Do you have any other symptoms, such as fainting, blurred vision, or weakness? I've had some blurred vision during the dizzy spells, but no fainting.Have you tried any treatments or medications? I tried drinking more water, thinking it might be dehydration, but it hasn't helped.Do you have any past medical history or chronic conditions? I have a history of anemia, but it was resolved a few years ago.Have you been hospitalized or had any surgeries? Yes, I had knee surgery following a sports injury five years ago.What about your family medical history? My father has heart disease, and my mother has thyroid issues.Are you aware of any allergies or hypersensitivities? I'm not aware of any allergies.Have you undergone any previous investigations or tests related to this issue? No, this is a new problem.Have you noticed any issues with other systems, such as your heart rate, digestion, or breathing? No, everything else seems fine.",
    "What is your name? My name is Emily Harris.What brings you in today? I've been having trouble sleeping for the past couple of months. I find it hard to fall asleep and often wake up in the middle of the night.Did anything specific trigger the insomnia, like stress or a change in routine? I started a new job recently, which has been stressful, but the sleep issues started even before that.Do you have any other symptoms, such as daytime fatigue, mood changes, or concentration issues? Yes, I feel tired during the day, and I've been more irritable than usual.Have you tried any treatments or medications? I've tried over-the-counter sleep aids, but they only help a little.Do you have any past medical history or chronic conditions? I have mild anxiety, which I manage with therapy.Have you been hospitalized or had any surgeries? No, I've never been hospitalized or had surgery.What about your family medical history? My mother has depression, and my father has hypertension.Are you aware of any allergies or hypersensitivities? I'm allergic to latex; it causes skin irritation.Have you undergone any previous investigations or tests related to this issue? No, I haven't.Have you noticed any issues with other systems, such as your digestion, heart, or lungs? No, everything else seems normal.",
    "What is your name? My name is Olivia Brown.What brings you in today? I've been experiencing stomach pain and bloating for the past two weeks. It's worse after eating.Did anything specific trigger the symptoms, like a new diet or illness? I recently returned from a trip abroad, but I didn't eat anything unusual that I can recall.Do you have any other symptoms, such as nausea, vomiting, or changes in bowel habits? Yes, I've had occasional diarrhea and feel nauseous after meals.Have you tried any treatments or medications? I've taken antacids and probiotics, but they haven't helped much.Do you have any past medical history or chronic conditions? I have a history of irritable bowel syndrome (IBS).Have you been hospitalized or had any surgeries? No, I've never been hospitalized or had surgery.What about your family medical history? My father has celiac disease, and my mother has type 1 diabetes.Are you aware of any allergies or hypersensitivities? I'm lactose intolerant.Have you undergone any previous investigations or tests related to this issue? No, not for this specific problem.Have you noticed any issues with other systems, such as your heart, lungs, or skin? No, everything else seems fine.",
    "What is your name? My name is James Lee.What brings you in today? I've been having a persistent cough for the past three weeks. It's dry and worse at night.Did anything specific trigger the cough, like an illness or exposure to irritants? I had a cold about a month ago, but the cough has lingered even after the other symptoms resolved.Do you have any other symptoms, such as fever, shortness of breath, or chest pain? No fever, but I feel slightly short of breath when the coughing fits are severe.Have you tried any treatments or medications? I've tried cough syrup, but it hasn't been very effective.Do you have any past medical history or chronic conditions? I have mild asthma, but it hasn't been a problem for years.Have you been hospitalized or had any surgeries? No, I've never been hospitalized or had surgery.What about your family medical history? My mother has asthma, and my father has high cholesterol.Are you aware of any allergies or hypersensitivities? I'm allergic to dust mites.Have you undergone any previous investigations or tests related to this issue? No, this is the first time I've had a cough last this long.Have you noticed any issues with other systems, such as your digestion, heart, or skin? No, everything else seems normal.",
    "What is your name? My name is Sophia Martinez.What brings you in today? I've been experiencing joint pain in my knees and wrists for the past month. It's worse in the morning and improves slightly as the day goes on.Did anything specific trigger the pain, like an injury or overuse? No, it started gradually without any clear cause.Do you have any other symptoms, such as swelling, stiffness, or fatigue? Yes, there's some swelling, and I feel unusually tired.Have you tried any treatments or medications? I've taken ibuprofen, which helps a bit with the pain but not the stiffness.Do you have any past medical history or chronic conditions? I have hypothyroidism, which I manage with medication.Have you been hospitalized or had any surgeries? Yes, I had gallbladder surgery three years ago.What about your family medical history? My mother has rheumatoid arthritis, and my father has hypertension.Are you aware of any allergies or hypersensitivities? I'm allergic to sulfa drugs.Have you undergone any previous investigations or tests related to this issue? No, this is the first time I've had joint problems.Have you noticed any issues with other systems, such as your heart, lungs, or digestion? No, everything else seems fine.",
    "What is your name? My name is Ethan Wilson.What brings you in today? I've been having a sore throat and swollen glands for about a week. It hurts to swallow, and my voice sounds hoarse.Did anything specific trigger the symptoms, like a recent illness or exposure to sick individuals? I work at a daycare, so I'm around kids who often have colds.Do you have any other symptoms, such as fever, cough, or fatigue? I had a mild fever for a couple of days, but no cough.Have you tried any treatments or medications? I've been drinking warm tea and taking over-the-counter pain relievers.Do you have any past medical history or chronic conditions? No, I'm generally healthy.Have you been hospitalized or had any surgeries? No, I've never been hospitalized or had surgery.What about your family medical history? My family doesn't have any major health issues.Are you aware of any allergies or hypersensitivities? No known allergies.Have you undergone any previous investigations or tests related to this issue? No, this is the first time I've had symptoms like this.Have you noticed any issues with other systems, such as your digestion, heart, or lungs? No, everything else seems normal.",
    "What is your name? My name is Isabella Davis.What brings you in today? I've been feeling unusually anxious and having frequent panic attacks over the past few weeks.Did anything specific trigger the anxiety, like stress or a traumatic event? I've been under work-related stress, but the anxiety feels out of proportion.Do you have any other symptoms, such as palpitations, shortness of breath, or chest pain? Yes, during the panic attacks, I feel my heart racing and it's hard to catch my breath.Have you tried any treatments or medications? I've tried deep breathing exercises, which help a little.Do you have any past medical history or chronic conditions? I have a history of mild depression.Have you been hospitalized or had any surgeries? No, I've never been hospitalized or had surgery.What about your family medical history? My mother has anxiety disorder, and my father has high blood pressure.Are you aware of any allergies or hypersensitivities? I'm allergic to peanuts.Have you undergone any previous investigations or tests related to this issue? No, this is a new problem.Have you noticed any issues with other systems, such as your digestion, heart, or skin? No, everything else seems fine.",
    "What is your name? My name is Liam Anderson.What brings you in today? I've been experiencing lower back pain for the past couple of weeks. It started after lifting a heavy box at work.Did anything specific trigger the pain, like an injury or overuse? Yes, it started immediately after lifting the box.Do you have any other symptoms, such as numbness, tingling, or weakness in your legs? No numbness or tingling, just the pain.Have you tried any treatments or medications? I've been using a heating pad and taking over-the-counter painkillers.Do you have any past medical history or chronic conditions? No, I'm generally healthy.Have you been hospitalized or had any surgeries? No, I've never been hospitalized or had surgery.What about your family medical history? My father has back problems related to a past injury.Are you aware of any allergies or hypersensitivities? I'm allergic to bee stings.Have you undergone any previous investigations or tests related to this issue? No, this is the first time I've had back pain like this.Have you noticed any issues with other systems, such as your digestion, heart, or lungs? No, everything else seems normal.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FLAN-T5 inference across consultation dialogues.")
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base", help="Model path or HF hub identifier.")
    parser.add_argument("--output_file", type=str, default="results/consultation_benchmark.txt", help="Path to save benchmark report.")
    parser.add_argument("--max_contexts", type=int, default=10, help="Number of consultation dialogues to benchmark.")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam count for generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initializing benchmark pipeline for %s...", args.model_name_or_path)

    pipeline = ClinicalQAPipeline(model_name_or_path=args.model_name_or_path)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    contexts = BENCHMARK_CONTEXTS[: args.max_contexts]
    latencies: List[float] = []

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== FLAN-T5 Inference Benchmark Report ===\n")
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Beams: {args.num_beams}, Total Contexts: {len(contexts)}\n")
        f.write("=" * 60 + "\n\n")

        for c_idx, context in enumerate(contexts):
            f.write(f"Consultation Context {c_idx + 1}:\n{context}\n")
            f.write("-" * 60 + "\n")
            logger.info("Processing consultation %d / %d...", c_idx + 1, len(contexts))

            for q_idx, (section_key, question) in enumerate(CLINICAL_QUESTIONS):
                res = pipeline.answer_question(
                    question=question,
                    context=context,
                    num_beams=args.num_beams,
                )
                latencies.append(res["latency_ms"])
                f.write(f"Q{q_idx + 1} [{section_key}]: {question}\n")
                f.write(f"A: {res['answer']}\n")
                f.write(f"Latency: {res['latency_ms']:.2f} ms\n\n")

            f.write("=" * 60 + "\n\n")

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        f.write(f"Benchmark Summary:\n")
        f.write(f"Total Queries: {len(latencies)}\n")
        f.write(f"Average Latency per Query: {avg_lat:.2f} ms\n")
        f.write(f"Min Latency: {min(latencies):.2f} ms | Max Latency: {max(latencies):.2f} ms\n")

    logger.info("Benchmark complete! Average latency: %.2f ms / query. Output saved to %s", avg_lat, out_path)


if __name__ == "__main__":
    main()

