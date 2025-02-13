# https://transcribe.intron.health/docs/?section=transcribe-with-api-file-queue
from pathlib import Path
import requests
import os

api_key = os.environ["INTRON_API_KEY"]


def upload_file(file_path: Path, api_key=api_key):
    url = "https://infer.intron.health/file/v1/upload"

    payload = {
        "audio_file_name": file_path.name
    }
    files = {
        'audio_file_blob': open(file_path, 'rb')
    }
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.request("POST", url, headers=headers, files=files, data=payload)
    response.raise_for_status()
    return response


def get_status(file_id, api_key=api_key):
    url = f"https://infer.intron.health/file/v1/status/{file_id}"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    DIRECTORY = Path("/Users/guillaumeallain/Documents/202501-consultation-recordings")
    filename = "sample.wav"
    response = upload_file(DIRECTORY / filename)
    file_id = response.json()["data"]["file_id"]

    print(get_status(file_id))
    # ... wait until data.processing_status: 'FILE_TRANSCRIBED'
    # {'data': {'audio_file_name': 'sample.wav', 'audio_transcript': "Hi there, doctor. I've had a fever for the past week with medium temperature. I've taken antibiotics, Amoxicillin 500mg in tablets, and I suspect I might have either a common cold or flu.", 'processed_audio_duration_in_seconds': 15, 'processing_status': 'FILE_TRANSCRIBED'}, 'message': 'file status found', 'status': 'Ok'}


