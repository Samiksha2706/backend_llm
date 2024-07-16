from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai

app = Flask(__name__)
CORS(app)  # Enable CORS

openai.api_key = ''

# Load the dataset
data_path = 'yh.csv'
df = pd.read_csv(data_path)

# Function to create the token count
def estimate_tokens(text):
    return len(text.split())

# Prepare the knowledge base
def prepare_knowledge_base(df, max_tokens=4096):
    knowledge_base = []
    for index, row in df.iterrows():
        hotel_info = (
            f"Hotel Name: {row['Hotel Names']}\n"
            f"Star Rating: {row['Star Rating']}\n"
            f"Rating: {row['Rating']}\n"
            f"Free Parking: {row['Free Parking']}\n"
            f"Fitness Centre: {row['Fitness Centre']}\n"
            f"Spa and Wellness Centre: {row['Spa and Wellness Centre']}\n"
            f"Airport Shuttle: {row['Airport Shuttle']}\n"
            f"Staff: {row['Staff']}\n"
            f"Facilities: {row['Facilities']}\n"
            f"Location: {row['Location']}\n"
            f"Comfort: {row['Comfort']}\n"
            f"Cleanliness: {row['Cleanliness']}\n"
            f"Price Per Day: ${row['Price Per Day($)']}\n"
        )
        knowledge_base.append(hotel_info)
    return "\n".join(knowledge_base)

knowledge_base = prepare_knowledge_base(df, max_tokens=3000)

# Define functions to get response and check relevance
def get_response(question, knowledge_base):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information about hotels."},
            {"role": "user", "content": f"{knowledge_base}\n\nQ: {question}\nA:"}
        ],
        #stream= True,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )
    answer = response.choices[0].message['content'].strip()
    return answer

def is_relevant(question, knowledge_base):
    # Convert user input to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # Define keywords based on your dataset columns and hotel names
    dataset_columns = [
        'hotel names', 'star rating', 'rating', 'free parking', 'fitness centre',
        'spa and wellness centre', 'airport shuttle', 'staff', 'facilities',
        'location', 'comfort', 'cleanliness', 'price per day'
    ]
    hotel_names_lower = df['Hotel Names'].str.lower().tolist()
    
    # Check if any dataset column keyword or hotel name is present in the question
    for column in dataset_columns:
        if column in question_lower:
            return True
    for hotel_name in hotel_names_lower:
        if hotel_name in question_lower:
            return True

    # If no specific keyword found, use OpenAI to verify relevance
    relevance_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that determines the relevance of questions based on the provided dataset."},
            {"role": "user", "content": f"Is the following question relevant to the provided dataset?\n\nDataset: {knowledge_base[:500]}...\n\nQuestion: {question}\n\nAnswer with 'Yes' or 'No':"}
        ],
        max_tokens=5,
        n=1,
        stop=None,
        temperature=0
    )
    relevance = relevance_response.choices[0].message['content'].strip()
    return relevance.lower() == 'yes'

# Handle user queries
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    print(f"Received question: {question}")  # Debugging log
    if is_relevant(question, knowledge_base):
        answer = get_response(question, knowledge_base)
        print(f"Answer: {answer}")  # Debugging log
        return jsonify({'answer': answer})
    else:
        print("Question is outside the scope of the dataset.")  # Debugging log
        return jsonify({'answer': "This is outside the scope of my dataset."})

if __name__ == '__main__':
    app.run(debug=True)