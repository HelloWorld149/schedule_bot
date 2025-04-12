# Web interface for the course scheduling chatbot

# Importing Libraries
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import json
import time
import os
from BotDefinition import OpenAIBot

# Create Flask app with session support
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session

# Initialize OpenAI bot with API key
chatbot = OpenAIBot("gpt-4o-mini", "your api key")

# Try to load the courses dataset
try:
    df = pd.read_excel("courses for CS.xlsx")
    # Columns needed for scheduling + professor info + grade columns
    col_names = [
        "Course Number", "Course Name", "Course Area", "Course Available",
        "Course Schedule", "Undergraudate/Graduate", "Professor Name",
        # Grade distribution columns
        "A", "A-", "A+", "AU", "B", "B-", "B+", "C", "C-", "C+", 
        "D", "D-", "D+", "E", "F"
    ]
    full_df = df[col_names].copy()
    
    # Create essential dataframe
    courses_df = full_df[[
        "Course Number", "Course Name", "Course Area", 
        "Course Available", "Course Schedule", "Undergraudate/Graduate"
    ]]
    
    # Flag to indicate successful data loading
    data_loaded = True
except Exception as e:
    print(f"Error loading course data: {e}")
    # Create empty dataframes as fallback
    full_df = pd.DataFrame()
    courses_df = pd.DataFrame()
    data_loaded = False

# Helper functions from mychatbot.py
def clean_json_output(output_str):
    """Remove triple-backtick fences from the model's output."""
    output_str = output_str.strip()
    if output_str.startswith("```"):
        lines = output_str.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        output_str = "\n".join(lines).strip()
    return output_str

def generate_schedule(instruction):
    """Generate a course schedule based on the instruction."""
    # Get available courses only
    available_courses = courses_df[courses_df["Course Available"] == "Y"]
    available_courses_csv = available_courses.to_csv(index=False)
    
    prompt = (
        "You are a course scheduler assistant AI.\n"
        "Below is a dataset of courses in CSV format. Only consider the courses that are available next semester:\n\n"
        f"{available_courses_csv}\n\n"
        "Using ONLY these available courses, please generate a course schedule that meets the following instruction:\n"
        f"{instruction}\n\n"
        "Make sure the schedule avoids any time conflicts (if possible, based on the 'Course Schedule' column). "
        "Return only the course schedule as a valid JSON array. Each element in the array should be a JSON object with the following keys: "
        "'Course Number', 'Course Name', 'Course Area', 'Course Schedule', 'Undergraudate/Graduate'."
    )
    
    # Reset conversation for a fresh context
    chatbot.reset_conversation()
    chatbot.update_system_prompt("You are a course scheduler assistant AI.")
    
    # Generate response
    response = chatbot.generate_response(prompt)
    return response

def parse_schedule_json(raw_output):
    """Parse the schedule JSON from the raw output."""
    cleaned = clean_json_output(raw_output)
    try:
        schedule_list = json.loads(cleaned)
        # Ensure it's a list
        if not isinstance(schedule_list, list):
            return []
        return schedule_list
    except json.JSONDecodeError:
        return []

def generate_grade_distribution(schedule_list):
    """Generate grade distribution data for the given schedule."""
    grade_columns = ["A", "A-", "A+", "AU", "B", "B-", "B+", "C", "C-", "C+", "D", "D-", "D+", "E", "F"]
    distributions = {}
    
    # GPA values for each grade
    gpa_values = {
        "A+": 4.0, "A": 4.0, "A-": 3.7,
        "B+": 3.3, "B": 3.0, "B-": 2.7,
        "C+": 2.3, "C": 2.0, "C-": 1.7,
        "D+": 1.3, "D": 1.0, "D-": 0.7,
        "F": 0.0, "E": 0.0
    }
    
    for course in schedule_list:
        course_num = str(course.get("Course Number", ""))
        row_match = full_df[full_df["Course Number"].astype(str) == course_num]
        if row_match.empty:
            distributions[course_num] = "Course not found"
        else:
            row = row_match.iloc[0]
            # Get individual grade distributions
            distribution = {g: float(row[g]) * 100 if pd.notna(row[g]) else 0 for g in grade_columns if g in row}
            print(distribution)
            # Calculate combined letter grade distributions
            combined_dist = {
                "A": sum(distribution.get(g, 0) for g in ["A+", "A", "A-"]),
                "B": sum(distribution.get(g, 0) for g in ["B+", "B", "B-"]),
                "C": sum(distribution.get(g, 0) for g in ["C+", "C", "C-"]),
                "D": sum(distribution.get(g, 0) for g in ["D+", "D", "D-"]),
                "F": distribution.get("F", 0) + distribution.get("E", 0)
            }
            print(combined_dist)
            
            # Calculate average GPA
            total_weighted_gpa = 0
            total_students = 0
            for grade, percentage in distribution.items():
                if grade != "AU" and grade in gpa_values:  # Skip AU (audit) grades
                    total_weighted_gpa += gpa_values[grade] * percentage
                    total_students += percentage
            
            avg_gpa = total_weighted_gpa / total_students if total_students > 0 else 0
            
            # Store all data
            distributions[course_num] = {
                "detailed": distribution,
                "combined": combined_dist,
                "avg_gpa": round(avg_gpa, 2)
            }
    
    return distributions

def ask_question_about_courses(question, schedule_list):
    """Ask a question about the courses or schedule."""
    dataset_csv = full_df.to_csv(index=False)
    schedule_json_str = json.dumps(schedule_list, indent=2)
    
    prompt = (
        "Here is the entire dataset of courses (including unavailable ones):\n"
        f"{dataset_csv}\n\n"
        "Here is the current selected schedule:\n"
        f"{schedule_json_str}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Please answer using the dataset or the schedule as needed. Make your response conversational, friendly and helpful."
    )
    
    # Reset conversation for a fresh context
    chatbot.reset_conversation()
    chatbot.update_system_prompt("You are a helpful course assistant.")
    
    # Generate response
    response = chatbot.generate_response(prompt)
    return response

def check_schedule_conflicts(schedule_list):
    """Check for time conflicts in the schedule."""
    time_slots = {}
    conflicts = []
    
    for course in schedule_list:
        schedule = course.get("Course Schedule", "")
        if schedule:
            if schedule in time_slots:
                conflicts.append({
                    "course1": course["Course Number"],
                    "course2": time_slots[schedule]["Course Number"],
                    "time": schedule
                })
            else:
                time_slots[schedule] = course
    
    return conflicts

def resolve_schedule_conflicts(instruction, schedule_list):
    """Try to generate a new schedule without conflicts."""
    if not schedule_list:
        return []
    
    conflict_resolution = f"{instruction} and make sure there are no time conflicts"
    raw_schedule_output = generate_schedule(conflict_resolution)
    new_schedule = parse_schedule_json(raw_schedule_output)
    return new_schedule

# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    # Initialize or reset the session data
    session['student_info'] = {}
    session['schedule'] = []
    session['instruction'] = ""
    return render_template('course_scheduler.html', data_loaded=data_loaded)

@app.route('/save_student_info', methods=['POST'])
def save_student_info():
    """Save student information to the session."""
    try:
        student_info = {
            'name': request.form.get('name', ''),
            'year': request.form.get('year', ''),
            'major': request.form.get('major', ''),
            'interests': request.form.get('interests', '')
        }
        session['student_info'] = student_info
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generate_schedule', methods=['POST'])
def create_schedule():
    """Generate a course schedule based on user instruction."""
    if not data_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Course data not loaded. Please check if "courses for CS.xlsx" exists.'
        })
    
    try:
        instruction = request.form.get('instruction', '')
        if not instruction:
            return jsonify({
                'status': 'error',
                'message': 'Please provide instructions for your schedule.'
            })
        
        # Store the instruction in session
        session['instruction'] = instruction
        
        # Generate the schedule
        raw_schedule_output = generate_schedule(instruction)
        schedule_list = parse_schedule_json(raw_schedule_output)
        
        if not schedule_list:
            # Try with simplified instruction
            student_info = session.get('student_info', {})
            year = student_info.get('year', 'undergraduate')
            interests = student_info.get('interests', 'computer science')
            
            simplified_instruction = f"Create a schedule with CS courses for a {year} student interested in {interests}"
            raw_schedule_output = generate_schedule(simplified_instruction)
            schedule_list = parse_schedule_json(raw_schedule_output)
            
            if not schedule_list:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not generate a valid schedule. Please try different instructions.'
                })
        
        # Store the schedule in session
        session['schedule'] = schedule_list
        
        # Check for conflicts
        conflicts = check_schedule_conflicts(schedule_list)
        
        # Get course areas for summary
        areas = list(set(course.get("Course Area", "Unknown") for course in schedule_list))
        
        return jsonify({
            'status': 'success',
            'schedule': schedule_list,
            'courseCount': len(schedule_list),
            'areas': areas,
            'conflicts': conflicts
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating schedule: {str(e)}'
        })

@app.route('/get_grade_distribution', methods=['POST'])
def get_grades():
    """Get grade distribution for the current schedule."""
    try:
        schedule_list = session.get('schedule', [])
        if not schedule_list:
            return jsonify({
                'status': 'error',
                'message': 'No schedule available. Please generate a schedule first.'
            })
        
        grade_dist = generate_grade_distribution(schedule_list)
        
        # Calculate insights about GPAs
        gpa_insights = []
        course_gpas = []
        
        for course_num, dist_data in grade_dist.items():
            if isinstance(dist_data, dict) and "avg_gpa" in dist_data:
                course_gpas.append((course_num, dist_data["avg_gpa"]))
        
        if course_gpas:
            course_gpas.sort(key=lambda x: x[1], reverse=True)
            highest_gpa = course_gpas[0]
            gpa_insights.append(f"Course {highest_gpa[0]} has the highest average GPA of {highest_gpa[1]:.2f}.")
            
            if len(course_gpas) > 1:
                lowest_gpa = course_gpas[-1]
                gpa_insights.append(f"Course {lowest_gpa[0]} has the lowest average GPA of {lowest_gpa[1]:.2f}.")
        
        # Calculate A-grade percentages for insights
        a_percentages = []
        for course_num, dist_data in grade_dist.items():
            if isinstance(dist_data, dict) and "combined" in dist_data:
                a_grade = dist_data["combined"]["A"]
                if a_grade > 0:
                    a_percentages.append((course_num, a_grade))
        
        insights = gpa_insights
        if a_percentages:
            a_percentages.sort(key=lambda x: x[1], reverse=True)
            highest_a = a_percentages[0]
            insights.append(f"Course {highest_a[0]} has the highest percentage of A grades ({highest_a[1]:.1f}%).")
            
            if len(a_percentages) > 1:
                lowest_a = a_percentages[-1]
                insights.append(f"Course {lowest_a[0]} has the lowest percentage of A grades ({lowest_a[1]:.1f}%).")
        
        return jsonify({
            'status': 'success',
            'grade_distribution': grade_dist,
            'insights': insights
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting grade distribution: {str(e)}'
        })

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Ask a question about courses or the schedule."""
    try:
        question = request.form.get('question', '')
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Please provide a question to ask.'
            })
        
        schedule_list = session.get('schedule', [])
        answer = ask_question_about_courses(question, schedule_list)
        
        return jsonify({
            'status': 'success',
            'answer': answer
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        })

@app.route('/resolve_conflicts', methods=['POST'])
def resolve_conflicts():
    """Try to resolve schedule conflicts."""
    try:
        schedule_list = session.get('schedule', [])
        instruction = session.get('instruction', '')
        
        if not schedule_list or not instruction:
            return jsonify({
                'status': 'error',
                'message': 'No schedule or instruction available.'
            })
        
        new_schedule = resolve_schedule_conflicts(instruction, schedule_list)
        
        if not new_schedule:
            return jsonify({
                'status': 'error',
                'message': 'Could not resolve all conflicts. Try manually removing some courses.'
            })
        
        # Update the session with the new schedule
        session['schedule'] = new_schedule
        
        # Check if there are still conflicts
        conflicts = check_schedule_conflicts(new_schedule)
        
        return jsonify({
            'status': 'success',
            'schedule': new_schedule,
            'conflicts': conflicts,
            'resolved': len(conflicts) == 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error resolving conflicts: {str(e)}'
        })

@app.route('/update_schedule', methods=['POST'])
def update_schedule():
    """Update the schedule with new instructions."""
    try:
        feedback = request.form.get('feedback', '')
        instruction = session.get('instruction', '')
        
        if not instruction:
            return jsonify({
                'status': 'error',
                'message': 'No previous instruction found. Please generate a schedule first.'
            })
        
        updated_instruction = instruction + " " + feedback
        raw_schedule_output = generate_schedule(updated_instruction)
        new_schedule = parse_schedule_json(raw_schedule_output)
        
        if not new_schedule:
            return jsonify({
                'status': 'error',
                'message': 'Could not update schedule with those requirements.'
            })
        
        # Update session data
        session['schedule'] = new_schedule
        session['instruction'] = updated_instruction
        
        # Check for conflicts
        conflicts = check_schedule_conflicts(new_schedule)
        
        return jsonify({
            'status': 'success',
            'schedule': new_schedule,
            'conflicts': conflicts
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error updating schedule: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)
