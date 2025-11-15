'''
Test script for the quiz generation API endpoint.

This script tests the new /api/v1/quiz/generate endpoint which:
1. Retrieves chunks by skill from Qdrant
2. Generates quiz questions using LLM
3. Generates explanations for each option in parallel

Usage:
    python test_quiz_api.py
'''

import requests
import json
import time

# API endpoint
API_URL = "http://localhost:8000"

def print_separator(title="", char="="):
    """Print a formatted separator line"""
    print("\n" + char * 80)
    if title:
        print(title)
        print(char * 80)

def test_health():
    """Test the health endpoint"""
    print_separator("Testing Health Endpoint")

    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_embed_status(collection_name="cs_materials"):
    """Check what skills are available in the collection"""
    print_separator("Checking Available Skills in Collection")

    try:
        response = requests.get(
            f"{API_URL}/api/v1/embed/status",
            params={"collection_name": collection_name}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Collection: {result['collection_name']}")
            print(f"Total Modules: {len(result['modules'])}")

            # Extract all unique skills
            all_skills = set()
            for module in result['modules']:
                for skill in module.get('skills', []):
                    all_skills.add(skill)

            if all_skills:
                print(f"\nAvailable Skills ({len(all_skills)}):")
                for i, skill in enumerate(sorted(all_skills), 1):
                    print(f"  {i}. {skill}")
                return list(all_skills)
            else:
                print("\nNo skills found in collection.")
                print("You may need to embed some documents first using the /api/v1/embed endpoint.")
                return []
        else:
            print(f"ERROR: {response.text}")
            return []

    except Exception as e:
        print(f"ERROR: {e}")
        return []

def test_quiz_generation(
    skill="Python programming basics",
    learning_objective="Introduction to Programming in Python",
    num_easy=1,
    num_medium=1,
    num_hard=1,
    question_style="concrete",
    collection_name="cs_materials"
):
    """Test the quiz generation endpoint"""
    print_separator("Testing Quiz Generation Endpoint")

    request_data = {
        "learning_objective": learning_objective,
        "skill": skill,
        "num_easy": num_easy,
        "num_medium": num_medium,
        "num_hard": num_hard,
        "question_style": question_style,
        "collection_name": collection_name
    }

    print(f"\nRequest:")
    print(json.dumps(request_data, indent=2))

    print("\nSending request to API...")
    print("(This may take a few minutes - LLM is generating questions and explanations)")

    start_time = time.time()

    try:
        response = requests.post(
            f"{API_URL}/api/v1/quiz/generate",
            json=request_data,
            timeout=600  # 10 minute timeout for LLM processing
        )

        elapsed_time = time.time() - start_time
        print(f"\nResponse received in {elapsed_time:.2f} seconds")
        print(f"Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            if result['success']:
                print_separator("SUCCESS!", "=")
                print(f"Generated {result['num_questions_generated']} questions")

                # Display each question with options and explanations
                for question in result['questions']:
                    print_separator(f"Question {question['id']}", "-")
                    print(f"Type: {question['type'].upper()}")
                    print(f"Difficulty: {question['difficulty'].upper()}")
                    print(f"Skill: {question['skill']}")
                    print(f"\n{question['question']}")

                    print(f"\nOptions:")
                    for option in question['options']:
                        correct_marker = "✓ CORRECT" if option['is_correct'] else "✗ Incorrect"
                        print(f"\n  [{option['id']}] {option['text']} - {correct_marker}")
                        print(f"      Explanation: {option['explanation']}")

                print_separator("Quiz Generation Complete", "=")

                # Summary statistics
                mcq_count = sum(1 for q in result['questions'] if q['type'] == 'mcq')
                tf_count = sum(1 for q in result['questions'] if q['type'] == 'true_false')
                easy_count = sum(1 for q in result['questions'] if q['difficulty'] == 'easy')
                medium_count = sum(1 for q in result['questions'] if q['difficulty'] == 'medium')
                hard_count = sum(1 for q in result['questions'] if q['difficulty'] == 'hard')

                print(f"\nSummary:")
                print(f"  Total Questions: {result['num_questions_generated']}")
                print(f"  MCQ Questions: {mcq_count}")
                print(f"  True/False Questions: {tf_count}")
                print(f"  Easy: {easy_count} | Medium: {medium_count} | Hard: {hard_count}")
                print(f"  Time Taken: {elapsed_time:.2f} seconds")

                return True
            else:
                print_separator("GENERATION FAILED", "=")
                print(f"Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print("\nERROR:")
            print(response.text)
            return False

    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out.")
        print("The LLM processing may take longer than expected.")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

def interactive_test():
    """Interactive mode - let user choose skill from available options"""
    print_separator("INTERACTIVE QUIZ GENERATION TEST", "=")

    # Check health first
    if not test_health():
        print("\nAPI is not healthy. Please check if the server is running.")
        return

    # Get available skills
    collection_name = input("\nEnter collection name (default: cs_materials): ").strip()
    if not collection_name:
        collection_name = "cs_materials"

    skills = test_embed_status(collection_name)

    if not skills:
        print("\nNo skills found. Please specify a skill manually.")
        skill = input("Enter skill name: ").strip()
        if not skill:
            print("No skill provided. Exiting.")
            return
    else:
        print("\nSelect a skill by number, or type a custom skill:")
        choice = input("Choice: ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(skills):
            skill = skills[int(choice) - 1]
        else:
            skill = choice

    # Get other parameters
    learning_objective = input("\nEnter learning objective (default: Course materials): ").strip()
    if not learning_objective:
        learning_objective = "Course materials"

    # Get difficulty distribution
    print("\nEnter the number of questions for each difficulty level:")
    num_easy_str = input("  Easy questions (default: 1): ").strip()
    num_easy = int(num_easy_str) if num_easy_str.isdigit() else 1

    num_medium_str = input("  Medium questions (default: 2): ").strip()
    num_medium = int(num_medium_str) if num_medium_str.isdigit() else 2

    num_hard_str = input("  Hard questions (default: 1): ").strip()
    num_hard = int(num_hard_str) if num_hard_str.isdigit() else 1

    # Get question style
    print("\nSelect explanation style:")
    print("  1. Concrete (example-driven with real-world scenarios)")
    print("  2. Abstract (technical terminology and formal definitions)")
    style_choice = input("Choice (1/2, default: 1): ").strip()
    question_style = "abstract" if style_choice == "2" else "concrete"

    # Generate quiz
    test_quiz_generation(
        skill=skill,
        learning_objective=learning_objective,
        num_easy=num_easy,
        num_medium=num_medium,
        num_hard=num_hard,
        question_style=question_style,
        collection_name=collection_name
    )

def automated_test():
    """Automated test with predefined parameters"""
    print_separator("AUTOMATED QUIZ GENERATION TEST", "=")

    # Test 1: Check health
    if not test_health():
        print("\nAPI is not healthy. Stopping tests.")
        return

    # Test 2: Check available skills
    skills = test_embed_status()

    # Test 3: Generate quiz with first available skill (or default)
    if skills:
        test_skill = skills[0]
        print(f"\nUsing first available skill: {test_skill}")
    else:
        test_skill = "Python programming basics"
        print(f"\nNo skills found. Using default skill: {test_skill}")
        print("Note: This may fail if no matching content exists in the collection.")

    test_quiz_generation(
        skill=test_skill,
        learning_objective="Introduction to Programming",
        num_easy=1,
        num_medium=1,
        num_hard=1,
        collection_name="cs_materials"
    )

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                    QUIZ GENERATION API TEST                            ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)

    print("Select test mode:")
    print("  1. Interactive mode (choose from available skills)")
    print("  2. Automated mode (test with first available skill)")
    print("  3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        interactive_test()
    elif choice == "2":
        automated_test()
    elif choice == "3":
        print("\nExiting.")
    else:
        print("\nInvalid choice. Running automated test...")
        automated_test()

    print_separator("DONE", "=")
