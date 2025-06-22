from typing import Tuple, Dict, List, Any
import whisper
from whisper.model import Whisper
from pyannote.audio import Pipeline as DiarizationPipeline
import torch
import logging
from pathlib import Path
from anthropic import AnthropicBedrock

import json
import ast
from time import time
from pydantic import BaseModel, Field

def load_models(diarization_model_auth_token: str) -> Tuple[Whisper, DiarizationPipeline]:
    """
    Load and return the transcription and diarization models.

    Args:
        diarization_model_auth_token (str): HuggingFace auth token for diarization model.

    Returns:
        Tuple[Whisper, DiarizationPipelineType]: (transcription_model, diarization_model)
    """
    transcription_model = whisper.load_model("base")
    diarization_model = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=diarization_model_auth_token
    ).to(torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    return transcription_model, diarization_model

# Transcription with timestamps
def transcribe(
    audio_path: Path,
    transcription_model: whisper.model,
    logging: = logging.Logger,
    word_timestamps: bool = False,
) -> Dict[str, str | List[Dict[str, object]] | None]:
    try:
        # Transcribe (get timestamps)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        result = transcription_model.transcribe(
            str(audio_path), word_timestamps=word_timestamps
        )
        return result
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return {}

# Speaker diarization
def diarize_audio(
    audio_path: Path, diarization_model: DiarizationPipeline, logging: logging.Logger
) -> list[tuple[float, float, str]]:
    try:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        diarization = diarization_model(audio_path)
        return [
            (segment.start, segment.end, label)
            for segment, _, label in diarization.itertracks(yield_label=True)
        ]
    except Exception as e:
        logging.error(f"Error during diarization: {e}")
        return []

def integrate_diarization_and_transcription(
    diarization: List[tuple[float, float, str]],
    transcription_chunks: List[Dict[str, object]] | None,
) -> list[dict[str, object]]:
    result = []

    for chunk in transcription_chunks:
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]
        text = chunk["text"]
        chunk_text = text.strip()
        if not chunk_text:
            continue
        best_match = None
        max_overlap = 0

        for turn_start, turn_end, speaker in diarization:
            # Calculate overlap
            overlap_start = max(chunk_start, turn_start)
            overlap_end = min(chunk_end, turn_end)
            overlap_duration = max(0.0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_match = (turn_start, turn_end, speaker)

        if best_match and max_overlap > 0:
            result.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "speaker": best_match[2],
                    "content": chunk_text,
                }
            )

    # Sort results by transcription start time
    result.sort(key=lambda x: x["start"])
    return result

def merge_consecutive_speaker_segments(
    segments: List[Dict[str, object]]
) -> str:
    """
    Merge consecutive speech segments by the same speaker into a single segment.

    Args:
        segments (List[Dict[str, object]]): A list of segments, each containing:
            - "start" (float): start time of the segment
            - "end" (float): end time of the segment
            - "speaker" (str): speaker label (e.g., "SPEAKER_00")
            - "content" (str): transcribed text of the segment

    Returns:
        List[Dict[str, object]]: A list of merged segments, where consecutive segments
        from the same speaker are combined. Each merged segment includes:
            - "start" (float): start time of the first segment in the group
            - "end" (float): end time of the last segment in the group
            - "speaker" (str): speaker label
            - "content" (str): concatenated content
    """
    if not segments:
        return []

    merged_segments: List[Dict[str, object]] = []
    current = {
        "speaker": segments[0]["speaker"],
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "content": segments[0]["content"],
    }

    for seg in segments[1:]:
        if seg["speaker"] == current["speaker"]:
            current["end"] = seg["end"]
            current["content"] += " " + seg["content"]
        else:
            merged_segments.append(current)
            current = {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "content": seg["content"],
            }

    merged_segments.append(current)
    merged_segments_json = json.dumps(merged_segments, indent=4)
    return merged_segments_json

def add_speakers_roles(transcription: str)-> str:
    transcription = ast.literal_eval(transcription)
    speakers_roles = {}
    for speaker in set([speaker["speaker"] for speaker in transcription]):
        speakers_roles[speaker] = get_speaker_role(speaker, transcription)
    for speaker in transcription:
        speaker["speaker_role"] = speakers_roles[speaker["speaker"]]
    transcription_with_speakers_roles = json.dumps(transcription, indent=4)
    return transcription_with_speakers_roles

def get_speaker_role(speaker: str, transcription: json) -> str:
    prompt = f"""
    You are given a JSON transcription of a conversation between multiple speakers, each labeled with a speaker name and their spoken content.

    Task: Based on the speaker's name and the context of the conversation, determine the speaker's role (e.g., Doctor, Patient, Nurse, etc.).
    If the role is unclear or cannot be inferred confidently, return "Unknown".

    The speaker's role should:
    - Be consistent throughout the conversation.
    - Be inferred from both the speaker's name and the conversational context.
    - Be returned **as a single string only**, with no explanations.

    The speaker name is: {speaker}
    The transcription is:
    {json.dumps(transcription)}
    """
    claude_client = AnthropicBedrock(aws_region="us-east-1")
    response = claude_client.messages.create(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        max_tokens=8192,
        system="",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.content[0].text

async def diarize_and_integrate(
    audio_path: Path,
    transcription_results: Dict[str, str | List[Dict[str, object]] | None],
    diarization_model: DiarizationPipeline,
    logging: logging.Logger,
) -> str:
    diarization_results = diarize_audio(audio_path, diarization_model, logging)
    integrated_data = integrate_diarization_and_transcription(
        diarization_results, transcription_results["segments"]
    )
    meeting_minutes = merge_consecutive_speaker_segments(integrated_data)
    meeting_minutes = add_speakers_roles(meeting_minutes)
    return meeting_minutes

async def transcription_and_diarization_main(audio_file_path: Path, diarization_model_auth_token: str) -> str:
    # Initialize models
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    transcription_model = whisper.load_model("base")

    auth_token = diarization_model_auth_token
    diarization_model = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=auth_token
    )
    diarization_model.to(device)
    # Step 1: Transcribe audio
    logging.info("Starting transcription...")
    start_time = time()
    transcription_results = transcribe(
        audio_file_path,
        transcription_model,
        logging.getLogger(),
        word_timestamps=True,
    )
    print(f"Transcription time: {time() - start_time:.2f} seconds")
    logging.info("Transcription completed.")

    # Step 2: Diarize and integrate
    logging.info("Starting diarization and integration...")
    start_time = time()
    integrated_results = await diarize_and_integrate(
        audio_file_path,
        transcription_results,
        diarization_model,
        logging.getLogger(),
    )
    print(f"Diarization and integration time: {time() - start_time:.2f} seconds")
    logging.info("Diarization and integration completed.")

    return integrated_results

# Medical Report Extraction and Analysis from Conversation

class ICD10Code(BaseModel):
    """
    Represents a single ICD-10 code and its description.
    """
    code: str = Field(..., description="The ICD-10 code")
    description: str = Field(..., description="The description of the ICD-10 code")


class ICD10Codes(BaseModel):
    """
    Represents a list of ICD-10 codes.
    """
    codes: List[ICD10Code] = Field(..., description="The ICD-10 codes")


class CPTCode(BaseModel):
    """
    Represents a single CPT code and its description.
    """
    code: str = Field(..., description="The CPT code")
    description: str = Field(..., description="The description of the CPT code")


class CPTCodes(BaseModel):
    """
    Represents a list of CPT codes.
    """
    codes: List[CPTCode] = Field(..., description="The CPT codes")


class ConversationReportAnalysis(BaseModel):
    """
    Represents the full analysis of a conversation report, including all relevant medical fields and codes.
    """
    chief_complaint: str = Field(..., description="The chief complaint of the patient")
    history_of_present_illness: str = Field(..., description="The history of present illness of the patient")
    past_medical_history: str = Field(..., description="The past medical history of the patient")
    past_surgical_history: str = Field(..., description="The past surgical history of the patient")
    medications: str = Field(..., description="The medications the patient is taking")
    allergies: str = Field(..., description="The allergies the patient has")
    family_history: str = Field(..., description="The family history of the patient")
    social_history: str = Field(..., description="The social history of the patient")
    review_of_systems: str = Field(..., description="The review of systems of the patient")
    neurological: str = Field(..., description="The neurological findings of the patient")
    psychiatric: str = Field(..., description="The psychiatric findings of the patient")
    endocrine: str = Field(..., description="The endocrine findings of the patient")
    hematologic_lymphatic: str = Field(..., description="The hematologic/lymphatic findings of the patient")
    allergic_immunologic: str = Field(..., description="The allergic/immunologic findings of the patient")
    physical_exam: str = Field(..., description="The physical exam of the patient")
    labs: str = Field(..., description="The labs of the patient")
    imaging: str = Field(..., description="The imaging of the patient")
    assessment_and_plan: str = Field(..., description="The assessment and plan of the patient")
    icd10_codes: ICD10Codes = Field(..., description="The ICD-10 codes of the patient")
    cpt_codes: CPTCodes = Field(..., description="The CPT codes of the patient. All codes relevant to the conversation should be suggested, it's ok to suggest codes you are not sure about")


def call_claude_with_tools(
    prompt: str,
    tools: List[Dict[str, Any]],
    client: Any,
    model: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
) -> Any:
    """
    Calls the Claude model with the provided prompt and tools.

    Args:
        prompt (str): The prompt to send to the model.
        tools (List[Dict[str, Any]]): The list of tools (schemas) for extraction.
        client (Any): The AnthropicBedrock client instance.
        model (str): The model identifier.

    Returns:
        Any: The response from the Claude model.
    """
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice={"type": "tool", "name": "analyse_conversation_report"},
        max_tokens=5000,
    )
    return response


def get_extraction_tools() -> List[Dict[str, Any]]:
    """
    Returns the extraction tools (schemas) for the Claude model.

    Returns:
        List[Dict[str, Any]]: The list of tool schemas.
    """
    tools = [
        {
            "name": "analyse_conversation_report",
            "description": "Analyze the conversation report.",
            "input_schema": ConversationReportAnalysis.schema(),
        },
    ]
    return tools


def load_conversation(conversation_file: str) -> str:
    """
    Loads the conversation text from a file.

    Args:
        conversation_file (str): Path to the conversation file.

    Returns:
        str: The conversation text.
    """
    with open(conversation_file, "r") as f:
        return f.read()


def create_report_from_conversation_report(conversation_report: Dict[str, Any]) -> str:
    """
    Creates a formatted report string from the conversation report dictionary.

    Args:
        conversation_report (Dict[str, Any]): The conversation report with titles and values.

    Returns:
        str: The formatted report string.
    """
    return f"""
{conversation_report['chief_complaint']['title']}:\n\n{conversation_report['chief_complaint']['value']}\n\n
{conversation_report['history_of_present_illness']['title']}:\n\n{conversation_report['history_of_present_illness']['value']}\n\n
{conversation_report['past_medical_history']['title']}:\n\n{conversation_report['past_medical_history']['value']}\n\n
{conversation_report['past_surgical_history']['title']}:\n\n{conversation_report['past_surgical_history']['value']}\n\n
{conversation_report['medications']['title']}:\n\n{conversation_report['medications']['value']}\n\n
{conversation_report['allergies']['title']}:\n\n{conversation_report['allergies']['value']}\n\n
{conversation_report['family_history']['title']}:\n\n{conversation_report['family_history']['value']}\n\n
{conversation_report['social_history']['title']}:\n\n{conversation_report['social_history']['value']}\n\n
{conversation_report['review_of_systems']['title']}:\n\n{conversation_report['review_of_systems']['value']}\n\n
{conversation_report['neurological']['title']}:\n\n{conversation_report['neurological']['value']}\n\n
{conversation_report['psychiatric']['title']}:\n\n{conversation_report['psychiatric']['value']}\n\n
{conversation_report['endocrine']['title']}:\n\n{conversation_report['endocrine']['value']}\n\n
{conversation_report['hematologic_lymphatic']['title']}:\n\n{conversation_report['hematologic_lymphatic']['value']}\n\n
{conversation_report['allergic_immunologic']['title']}:\n\n{conversation_report['allergic_immunologic']['value']}\n\n
{conversation_report['physical_exam']['title']}:\n\n{conversation_report['physical_exam']['value']}\n\n
{conversation_report['labs']['title']}:\n\n{conversation_report['labs']['value']}\n\n
{conversation_report['imaging']['title']}:\n\n{conversation_report['imaging']['value']}\n\n
{conversation_report['assessment_and_plan']['title']}:\n\n{conversation_report['assessment_and_plan']['value']}\n\n
"""


def extract_report_from_conversation(conversation_text: str) -> Dict[str, Any]:
    """
    Extracts a structured report from the conversation text using the Claude model.

    Args:
        conversation_text (str): The conversation transcript.

    Returns:
        Dict[str, Any]: The structured report with titles and values.
    """
    tools = get_extraction_tools()
    client = AnthropicBedrock(aws_region="us-east-2",)
    prompt = f"""
    You are a medical doctor, writting a medical report based on a conversation transcript.
    Given the following conversation, extract the following feilds according to their structure guidelines if appears.

    The fields are:
    Chief Complaint (CC)
    History of Present Illness (HPI) - Structure: Gender, Age, Relevant history titles, Chronological order of events, Substantive content
    Past Medical History (PMH)
    Past Surgical History (PSH)
    Medications
    Allergies
    Family History
    Social History
    Review of Systems (ROS)
    Neurological
    Psychiatric
    Endocrine
    Hematologic/Lymphatic
    Allergic/Immunologic
    Physical Exam
    Labs
    Imaging
    Assessment and Plan - Structure: One or two sentences summary, Assessment, Plan
    ICD-10 Codes
    CPT Codes (Suggest codes that you think are relevant to the conversation, it's ok to suggest codes you are not sure about)

    For icd10_codes and cpt_codes, return results as dictionaries! Our lives depend on it!

    The conversation transcript:
    {conversation_text}
    """
    time_start = time()
    response = call_claude_with_tools(prompt, tools, client)
    time_end = time()
    print(f"Claude call time: {time_end - time_start} seconds")

    model_output = response.content[0].input

    if not isinstance(model_output.get("icd10_codes"), dict):
        model_output["icd10_codes"] = {"codes": model_output["icd10_codes"]}

    if not isinstance(model_output.get("cpt_codes"), dict):
        model_output["cpt_codes"] = {"codes": model_output["cpt_codes"]}

    report_parameters = ConversationReportAnalysis(**model_output)
    report_parameters_with_titles = add_titles_to_report(report_parameters)
    return report_parameters_with_titles


def add_titles_to_report(report: ConversationReportAnalysis) -> Dict[str, Dict[str, Any]]:
    """
    Adds titles and descriptions to each field in the report.

    Args:
        report (ConversationReportAnalysis): The report object.

    Returns:
        Dict[str, Dict[str, Any]]: The report with titles, descriptions, and values.
    """
    schema = report.schema()
    values = report.dict()
    detailed_report = {
        field: {
            "title": schema["properties"][field].get("title", ""),
            "description": schema["properties"][field].get("description", ""),
            "value": values[field]
        }
        for field in values
    }
    return detailed_report


def transcription_to_report_main(conversation_file: str, report_file: str):
    """
    Main function to extract a report from a conversation file and save it.

    Args:
        conversation_file (str): Path to the conversation transcript file.
        report_file (str): Path to save the generated report.
    """
    conversation_text = load_conversation(conversation_file)
    report_parameters = extract_report_from_conversation(conversation_text)
    with open(report_file.replace(".txt", ".json"), "w") as f:
        json.dump(report_parameters, f, indent=4)
    report_text = create_report_from_conversation_report(report_parameters)
    with open(report_file, "w") as f:
        f.write(report_text)
