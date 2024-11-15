#import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from flask import Flask
from flask import Flask, jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up Groq API key
groq_api_key = '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'

@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    # Extract data from the request
    data = request.json
    ingredients = data.get('ingredients', '')
    model = data.get('model', 'llama3-8b-8192')

    # Set up conversation memory and Groq model
    memory = ConversationBufferWindowMemory(
        k=3, memory_key="chat_history", return_messages=True
    )
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # Define the system message for recipe generation
    system_prompt = SystemMessage(content=(
        "Generate a creative and unique recipe using only the following ingredients. "
        "Suggest a name for the recipe and the cooking method. No other ingredients should be used."
    ))

    # Set up prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            HumanMessagePromptTemplate.from_template("Ingredients: {ingredients}")
        ]
    )

    # Create the Langchain conversation chain
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        memory=memory,
    )

    # Generate recipe using the conversation chain
    try:
        recipe_suggestion = conversation.predict(ingredients=ingredients)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"recipe": recipe_suggestion})

if __name__ == "__main__":
    app.run(debug=True)
