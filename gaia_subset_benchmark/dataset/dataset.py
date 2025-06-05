import requests

# Get all questions
response = requests.get("https://agents-course-unit4-scoring.hf.space/questions")
questions_data = response.json()
print(f"Downloaded {len(questions_data)} questions")

# Save to file
import json
with open("gaia_questions.json", "w") as f:
    json.dump(questions_data, f, indent=2)

import requests

# First get questions to see available task_ids
questions_response = requests.get("https://agents-course-unit4-scoring.hf.space/questions")
questions = questions_response.json()

# Download files for each task
for question in questions:
    task_id = question.get("task_id")
    if task_id:
        try:
            file_response = requests.get(f"https://agents-course-unit4-scoring.hf.space/files/{task_id}")
            if file_response.status_code == 200:
                with open(f"task_{task_id}_file", "wb") as f:
                    f.write(file_response.content)
                print(f"Downloaded file for task {task_id}")
        except Exception as e:
            print(f"No file or error for task {task_id}: {e}")