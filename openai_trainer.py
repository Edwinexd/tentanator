"""
OpenAI Fine-tuning Module for Tentanator
Handles uploading training data and managing fine-tuning jobs
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import dotenv
from openai import OpenAI

dotenv.load_dotenv()


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning job"""
    model: str = "gpt-4.1-mini-2025-04-14"  # Default model for fine-tuning
    n_epochs: int = 3  # Number of training epochs
    batch_size: int = 1  # Batch size for training
    learning_rate_multiplier: float = 1.0
    suffix: Optional[str] = None  # Custom suffix for the model name
    validation_file: Optional[str] = None  # Optional validation dataset


@dataclass
class TrainingFile:
    """Represents an uploaded training file"""
    file_id: str
    filename: str
    question_name: str
    size: int
    created_at: str


@dataclass
class FineTuningJob:
    """Represents a fine-tuning job"""
    job_id: str
    model: str
    status: str
    created_at: str
    training_file: str  # JSONL file used for training
    question_name: str  # Question being trained
    exam_id: str = ""  # Exam identifier
    finished_at: Optional[str] = None
    fine_tuned_model: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ModelRegistry:
    """Registry of fine-tuned models"""
    models: Dict[str, Dict[str, Any]]  # model_id -> model info

    def add_model(self, model_id: str, jsonl_file: str, question_name: str, job_id: str, exam_id: str = ""):
        """Add a model to the registry"""
        self.models[model_id] = {
            "model_id": model_id,
            "jsonl_file": jsonl_file,
            "question_name": question_name,
            "exam_id": exam_id,  # Which exam this model was trained on
            "job_id": job_id,
            "created_at": datetime.now().isoformat()
        }

    def save(self, filepath: str = "models.json"):
        """Save registry to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=2)

    @classmethod
    def load(cls, filepath: str = "models.json") -> "ModelRegistry":
        """Load registry from file"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls(models=data)
        return cls(models={})


class OpenAITrainer:
    """Manages OpenAI fine-tuning for grading models"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            msg = "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            raise ValueError(msg)

        self.client = OpenAI(api_key=self.api_key)
        self.session_file = ".tentanator_training_session.json"
        self.model_registry = ModelRegistry.load()

    def validate_jsonl_file(self, filepath: Path) -> Tuple[bool, str, int]:
        """
        Validate JSONL file format for OpenAI fine-tuning
        Returns: (is_valid, message, num_examples)
        """
        if not filepath.exists():
            return False, f"File not found: {filepath}", 0

        examples = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        example = json.loads(line)
                        examples.append(example)

                        # Validate structure
                        if "messages" not in example:
                            return False, f"Line {line_num}: Missing 'messages' field", 0

                        messages = example["messages"]
                        if not isinstance(messages, list) or len(messages) < 2:
                            return False, f"Line {line_num}: 'messages' must be a list with at least 2 items", 0

                        # Check for required roles
                        roles = [msg.get("role") for msg in messages]
                        if "system" not in roles and "user" not in roles:
                            return False, f"Line {line_num}: Messages must include 'system' or 'user' role", 0

                        if "assistant" not in roles:
                            return False, f"Line {line_num}: Messages must include 'assistant' role", 0

                    except json.JSONDecodeError as e:
                        return False, f"Line {line_num}: Invalid JSON - {e}", 0

        except (OSError, json.JSONDecodeError) as e:
            return False, f"Error reading file: {e}", 0

        # Check minimum examples (OpenAI recommends at least 10)
        if len(examples) < 10:
            msg = f"Warning: Only {len(examples)} examples found. OpenAI recommends at least 10 for effective fine-tuning."
            return False, msg, len(examples)

        return True, f"Valid: {len(examples)} training examples", len(examples)

    def upload_training_file(self, filepath: Path, question_name: str) -> Optional[TrainingFile]:
        """Upload JSONL file to OpenAI for fine-tuning"""
        # Validate file first
        is_valid, message, num_examples = self.validate_jsonl_file(filepath)
        if not is_valid and num_examples == 0:
            print(f"❌ Validation failed: {message}")
            return None
        if not is_valid:
            print(f"⚠️  {message}")
            proceed = input("Continue anyway? [y/n]: ").strip().lower()
            if proceed != 'y':
                return None

        # Create a temporary file with "exam" removed from the filename for OpenAI
        temp_filepath = filepath.parent / filepath.name.replace("exam", "").replace("Exam", "")

        # If no change needed, use original file
        if temp_filepath == filepath:
            temp_filepath = filepath
            print(f"📤 Uploading {filepath.name} ({num_examples} examples)...")
        else:
            # Copy to temp file with new name
            shutil.copy2(filepath, temp_filepath)
            print(f"📤 Uploading as {temp_filepath.name} ({num_examples} examples)...")

        try:
            with open(temp_filepath, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )

            training_file = TrainingFile(
                file_id=response.id,
                filename=filepath.name,  # Keep original filename for tracking
                question_name=question_name,
                size=filepath.stat().st_size,
                created_at=datetime.now().isoformat()
            )

            # Clean up temporary file if we created one
            if temp_filepath != filepath and temp_filepath.exists():
                temp_filepath.unlink()

            print(f"✅ Uploaded successfully! File ID: {response.id}")
            return training_file

        except (OSError, RuntimeError, ValueError) as e:
            print(f"❌ Upload failed: {e}")
            # Clean up temporary file on error
            if temp_filepath != filepath and temp_filepath.exists():
                temp_filepath.unlink()
            return None

    def create_fine_tuning_job(self, training_file: TrainingFile,
                              jsonl_filename: str,
                              config: FineTuningConfig,
                              exam_id: str = "") -> Optional[FineTuningJob]:
        """Create a fine-tuning job with uploaded file"""
        print(f"🚀 Creating fine-tuning job for {training_file.question_name}...")

        try:
            # Generate suffix and remove "exam" from it
            if config.suffix:
                suffix = config.suffix.replace("exam", "").replace("Exam", "")
            else:
                suffix = f"tentanator_{training_file.question_name.replace(' ', '_').lower()}"
                suffix = suffix.replace("exam", "").replace("Exam", "")

            # Create job
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file.file_id,
                model=config.model,
                suffix=suffix
            )

            job = FineTuningJob(
                job_id=response.id,
                model=config.model,
                status=response.status,
                created_at=datetime.now().isoformat(),
                training_file=jsonl_filename,
                question_name=training_file.question_name,
                exam_id=exam_id
            )

            print(f"✅ Fine-tuning job created! Job ID: {response.id}")
            print(f"   Status: {response.status}")
            return job

        except (OSError, RuntimeError, ValueError) as e:
            print(f"❌ Failed to create job: {e}")
            return None

    def check_job_status(self, job: FineTuningJob) -> FineTuningJob:
        """Check the status of a fine-tuning job"""
        try:
            response = self.client.fine_tuning.jobs.retrieve(job.job_id)

            job.status = response.status
            job.fine_tuned_model = response.fine_tuned_model

            if response.status == "succeeded":
                job.finished_at = datetime.now().isoformat()
                print(f"✅ Job {job.job_id} completed successfully!")
                print(f"   Fine-tuned model: {job.fine_tuned_model}")

                # Add to model registry
                if job.fine_tuned_model:
                    self.model_registry.add_model(
                        model_id=job.fine_tuned_model,
                        jsonl_file=job.training_file,
                        question_name=job.question_name,
                        job_id=job.job_id,
                        exam_id=job.exam_id
                    )
                    self.model_registry.save()
                    print("   Added to models.json registry")

            elif response.status == "failed":
                job.error = response.error.message if response.error else "Unknown error"
                print(f"❌ Job {job.job_id} failed: {job.error}")
            else:
                print(f"⏳ Job {job.job_id} status: {response.status}")

            return job

        except (OSError, RuntimeError, ValueError) as e:
            print(f"❌ Failed to check job status: {e}")
            return job

    def monitor_job(self, job: FineTuningJob, check_interval: int = 30) -> FineTuningJob:
        """Monitor a fine-tuning job until completion"""
        print(f"📊 Monitoring job {job.job_id}...")
        print("   This may take 10-30 minutes depending on dataset size")

        while job.status not in ["succeeded", "failed", "cancelled"]:
            time.sleep(check_interval)
            job = self.check_job_status(job)

            # Show progress events
            try:
                events = self.client.fine_tuning.jobs.list_events(
                    fine_tuning_job_id=job.job_id,
                    limit=5
                )
                for event in events.data[:1]:  # Show latest event
                    print(f"   [{event.created_at}] {event.message}")
            except (OSError, RuntimeError, ValueError, AttributeError):
                pass

        return job

    def test_fine_tuned_model(self, model_name: str, test_input: str) -> Optional[str]:
        """Test a fine-tuned model with sample input"""
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are grading student responses."},
                    {"role": "user", "content": test_input}
                ],
                max_tokens=10,
                temperature=0.1
            )

            return response.choices[0].message.content

        except (OSError, RuntimeError, ValueError) as e:
            print(f"❌ Failed to test model: {e}")
            return None

    def batch_grade_with_model(self, model_name: str, csv_data: List[Dict[str, str]],
                              input_column: str, output_column: str) -> List[Dict[str, str]]:
        """Grade multiple responses using fine-tuned model"""
        print(f"🤖 Batch grading with {model_name}...")
        graded_results = []

        for idx, row in enumerate(csv_data):
            response_text = row.get(input_column, "")

            # Skip blank responses
            if response_text.strip() in ["", "-", "N/A"]:
                graded_results.append({
                    "row_index": idx,
                    "input": response_text,
                    "grade": "0",
                    "confidence": "auto"
                })
                continue

            try:
                # Get grade from model
                result = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": f"You are grading responses for: {output_column}"},
                        {"role": "user", "content": response_text}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )

                grade = result.choices[0].message.content.strip() # type: ignore

                graded_results.append({
                    "row_index": idx,
                    "input": response_text[:100] + "..." if len(response_text) > 100 else response_text,
                    "grade": grade,
                    "confidence": "model"
                })

                # Progress indicator
                if (idx + 1) % 10 == 0:
                    print(f"   Graded {idx + 1}/{len(csv_data)} responses...")

            except (OSError, RuntimeError, ValueError) as e:
                print(f"   Error grading row {idx}: {e}")
                graded_results.append({
                    "row_index": idx,
                    "input": response_text[:100] + "..." if len(response_text) > 100 else response_text,
                    "grade": "ERROR",
                    "confidence": "failed"
                })

        print(f"✅ Graded {len(graded_results)} responses")
        return graded_results

    def save_training_session(self, files: List[TrainingFile], jobs: List[FineTuningJob]):
        """Save training session to file"""
        session_data = {
            "last_updated": datetime.now().isoformat(),
            "uploaded_files": [asdict(f) for f in files],
            "fine_tuning_jobs": [asdict(j) for j in jobs]
        }

        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)

        print(f"💾 Session saved to {self.session_file}")

    def load_training_session(self) -> Tuple[List[TrainingFile], List[FineTuningJob]]:
        """Load training session from file and clean up cancelled jobs"""
        if not Path(self.session_file).exists():
            return [], []

        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            files = [TrainingFile(**f) for f in data.get("uploaded_files", [])]
            all_jobs = [FineTuningJob(**j) for j in data.get("fine_tuning_jobs", [])]

            # Filter out cancelled jobs
            active_jobs = []
            for job in all_jobs:
                if job.status == "cancelled":
                    print(f"🗑️  Removing cancelled job: {job.job_id} ({job.question_name})")
                else:
                    active_jobs.append(job)

            # Save cleaned session if any jobs were removed
            if len(active_jobs) < len(all_jobs):
                self.save_training_session(files, active_jobs)
                print(f"✅ Cleaned up {len(all_jobs) - len(active_jobs)} cancelled jobs")

            return files, active_jobs

        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Failed to load session: {e}")
            return [], []


def main():
    """Example usage and testing"""
    print("=== OpenAI Fine-Tuning Manager ===\n")

    # Initialize trainer
    trainer = OpenAITrainer()

    # Show existing models
    if trainer.model_registry.models:
        print("📚 Registered Models:")
        for model_id, info in trainer.model_registry.models.items():
            print(f"  - {info['question_name']}: {model_id}")
            print(f"    JSONL: {info['jsonl_file']}")

    # Load existing session
    files, jobs = trainer.load_training_session()

    if jobs:
        print("\n📋 Existing fine-tuning jobs:")
        updated_jobs = []
        for job in jobs:
            print(f"  - {job.question_name} ({job.job_id}): {job.status}")
            if job.status in ["running", "created", "validating_files"]:
                # Check current status
                updated_job = trainer.check_job_status(job)
                job.status = updated_job.status
                job.fine_tuned_model = updated_job.fine_tuned_model

            # Only keep non-cancelled jobs
            if job.status != "cancelled":
                updated_jobs.append(job)
            else:
                print("    🗑️  Removing cancelled job")

        if len(updated_jobs) != len(jobs):
            jobs = updated_jobs
            trainer.save_training_session(files, jobs)

    # Find JSONL files
    training_dir = Path("training_data")
    if training_dir.exists():
        jsonl_files = list(training_dir.glob("*.jsonl"))

        if jsonl_files:
            print("\n📁 Available JSONL files:")
            for idx, f in enumerate(jsonl_files, 1):
                # Check if already used
                already_used = any(
                    info['jsonl_file'] == f.name
                    for info in trainer.model_registry.models.values()
                )
                status = " ✅ (already trained)" if already_used else ""
                print(f"  {idx}. {f.name}{status}")

            # Select one file for training
            untrained_files = [f for f in jsonl_files
                              if not any(info['jsonl_file'] == f.name
                                       for info in trainer.model_registry.models.values())]

            if untrained_files:
                choice = input("\nSelect a file to train (enter number, or 'q' to quit): ").strip()

                if choice.lower() != 'q':
                    try:
                        file_idx = int(choice) - 1
                        if 0 <= file_idx < len(jsonl_files):
                            jsonl_file = jsonl_files[file_idx]

                            # Double-check if already trained
                            if any(info['jsonl_file'] == jsonl_file.name
                                  for info in trainer.model_registry.models.values()):
                                print(f"❌ {jsonl_file.name} has already been trained!")
                                print("   Model exists in registry. Cannot train again.")
                                return

                            # Check if there's an active job for this file
                            active_job = next((j for j in jobs
                                             if j.training_file == jsonl_file.name
                                             and j.status in ["running", "created", "validating_files"]),
                                            None)
                            if active_job:
                                print(f"❌ {jsonl_file.name} already has an active training job!")
                                print(f"   Job {active_job.job_id} is {active_job.status}")
                                return

                            # Extract question name from filename
                            question_name = jsonl_file.stem.rsplit('_', 2)[0]

                            # Upload file
                            print(f"\n📤 Processing {jsonl_file.name}")
                            training_file = trainer.upload_training_file(jsonl_file, question_name)
                            if training_file:
                                files.append(training_file)

                                # Extract exam_id from filename (format: examid_question_timestamp.jsonl)
                                filename_parts = jsonl_file.stem.split('_')
                                exam_id = filename_parts[0] if len(filename_parts) > 2 else ""

                                # Create fine-tuning job
                                config = FineTuningConfig()
                                job = trainer.create_fine_tuning_job(
                                    training_file, jsonl_file.name, config, exam_id
                                )
                                if job:
                                    jobs.append(job)

                                    # Monitor job
                                    monitor = input("Monitor job until completion? [y/n]: ").strip().lower()
                                    if monitor == 'y':
                                        job = trainer.monitor_job(job)

                                # Save session
                                trainer.save_training_session(files, jobs)
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Invalid input")
            else:
                print("\n✅ All files have been trained!")


if __name__ == "__main__":
    main()
