'''
Test script for the reading material generation API endpoint.

This script tests the new /api/v1/reading/generate endpoint which:
1. Retrieves chunks by skill from Qdrant
2. Generates educational reading material in markdown format using LLM
3. Supports both concrete (example-driven) and abstract (technical) styles

Usage:
    python test_reading_api.py
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

def test_reading_generation(
    skill="Python programming basics",
    learning_objective="Introduction to Programming in Python",
    mastery_level="beginner",
    collection_name="cs_materials",
    save_to_file=True
):
    """Test the reading material generation endpoint"""
    print_separator("Testing Reading Material Generation Endpoint")

    request_data = {
        "learning_objective": learning_objective,
        "skill": skill,
        "mastery_level": mastery_level,
        "collection_name": collection_name
    }

    print(f"\nRequest:")
    print(json.dumps(request_data, indent=2))

    print("\nSending request to API...")
    print("(This may take a few minutes - LLM is generating content)")

    start_time = time.time()

    try:
        response = requests.post(
            f"{API_URL}/api/v1/reading/generate",
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
                content = result['content']
                content_length = len(content)
                word_count = len(content.split())
                line_count = len(content.split('\n'))

                print(f"Generated reading material:")
                print(f"  - Characters: {content_length}")
                print(f"  - Words: {word_count}")
                print(f"  - Lines: {line_count}")
                print(f"  - Time Taken: {elapsed_time:.2f} seconds")

                # Save to file if requested
                if save_to_file:
                    # Create safe filename from skill name
                    safe_skill = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in skill)
                    safe_skill = safe_skill.replace(' ', '_').lower()
                    filename = f"reading_{safe_skill}_{mastery_level}.md"

                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)

                    print(f"\n✓ Content saved to: {filename}")

                # Display preview (first 1000 characters)
                print_separator("Content Preview (first 1000 chars)", "-")
                print(content[:1000])
                if len(content) > 1000:
                    print("\n... (content truncated in preview)")

                print_separator("Reading Material Generation Complete", "=")

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
    print_separator("INTERACTIVE READING MATERIAL TEST", "=")

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

    # Get mastery level
    print("\nSelect mastery level:")
    print("  1. Beginner (simple language, inline definitions, no jargon)")
    print("  2. Intermediate (technical depth with accessibility)")
    print("  3. Advanced (full technical rigor, industry terminology)")
    mastery_choice = input("Choice (1/2/3, default: 1): ").strip()
    if mastery_choice == "2":
        mastery_level = "intermediate"
    elif mastery_choice == "3":
        mastery_level = "advanced"
    else:
        mastery_level = "beginner"

    # Ask if user wants to save to file
    save_choice = input("\nSave content to markdown file? (y/n, default: y): ").strip().lower()
    save_to_file = save_choice != 'n'

    # Generate reading material
    test_reading_generation(
        skill=skill,
        learning_objective=learning_objective,
        mastery_level=mastery_level,
        collection_name=collection_name,
        save_to_file=save_to_file
    )

def automated_test():
    """Automated test with predefined parameters"""
    print_separator("AUTOMATED READING MATERIAL TEST", "=")

    # Test 1: Check health
    if not test_health():
        print("\nAPI is not healthy. Stopping tests.")
        return

    # Test 2: Check available skills
    skills = test_embed_status()

    # Test 3: Generate reading material with first available skill (or default)
    if skills:
        test_skill = skills[0]
        print(f"\nUsing first available skill: {test_skill}")
    else:
        test_skill = "Python programming basics"
        print(f"\nNo skills found. Using default skill: {test_skill}")
        print("Note: This may fail if no matching content exists in the collection.")

    # Test all mastery levels
    print_separator("Testing BEGINNER Level", "=")
    test_reading_generation(
        skill=test_skill,
        learning_objective="Introduction to Programming",
        mastery_level="beginner",
        collection_name="cs_materials",
        save_to_file=True
    )

    print("\n\n")

    print_separator("Testing INTERMEDIATE Level", "=")
    test_reading_generation(
        skill=test_skill,
        learning_objective="Introduction to Programming",
        mastery_level="intermediate",
        collection_name="cs_materials",
        save_to_file=True
    )

    print("\n\n")

    print_separator("Testing ADVANCED Level", "=")
    test_reading_generation(
        skill=test_skill,
        learning_objective="Introduction to Programming",
        mastery_level="advanced",
        collection_name="cs_materials",
        save_to_file=True
    )

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║              READING MATERIAL GENERATION API TEST                      ║
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
