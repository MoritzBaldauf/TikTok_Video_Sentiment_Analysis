import pandas as pd
import os
import json

# Function to extract data from Excel and save to JSON
def extract_and_save_json(file_name):
    # Load Excel file
    df = pd.read_csv(file_name)
    
    # Create JSON structure
    posts = []
    for index, row in df.iterrows():
        post = {
            "user_id": index + 1,
            "profile": "/images/propic.jpg",
            "like": row["Likes"],
            "comments": row["Comments"],
            "bookmarks": row["Saved"],
            "shares": row["Shares"],
            "description": row["Description"] if pd.notna(row["Description"]) else "",
            "political_bias_category": row["political_bias_category"],
            "religious_bias_category": row["religious_bias_category"],
            "overall_bias_category": row["overall_bias_category"]
        }
        posts.append(post)
    
    # Locate 'data' directory
    data_directory = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_directory):
        print(f"Directory '{data_directory}' not found")
        return

    # Save JSON to file
    json_file = os.path.join(data_directory, "posts.json")
    with open(json_file, "w") as f:
        json.dump(posts, f, indent=4)
    print(f"JSON data saved to {json_file}")

if __name__ == "__main__":
    extract_and_save_json("results_API.csv")
